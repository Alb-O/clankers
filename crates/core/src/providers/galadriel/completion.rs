use serde::{Deserialize, Serialize};
use tracing::{Instrument, enabled, info_span};

use super::client::{Client, Galadriel};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::message::MessageError;
use crate::providers::openai;
use crate::providers::openai_compat::{self, FlatApiError};
use crate::streaming::StreamingCompletionResponse;
use crate::{json_utils, message};

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
	pub role: String,
	pub content: Option<String>,
	#[serde(default, deserialize_with = "json_utils::null_or_vec")]
	pub tool_calls: Vec<openai::ToolCall>,
}

impl Message {
	fn system(preamble: &str) -> Self {
		Self {
			role: "system".to_string(),
			content: Some(preamble.to_string()),
			tool_calls: Vec::new(),
		}
	}
}

impl TryFrom<message::Message> for Message {
	type Error = message::MessageError;

	fn try_from(message: message::Message) -> Result<Self, Self::Error> {
		match message {
			message::Message::User { content } => Ok(Self {
				role: "user".to_string(),
				content: content.iter().find_map(|c| match c {
					message::UserContent::Text(text) => Some(text.text.clone()),
					_ => None,
				}),
				tool_calls: vec![],
			}),
			message::Message::Assistant { content, .. } => {
				let mut text_content: Option<String> = None;
				let mut tool_calls = vec![];

				for c in content.iter() {
					match c {
						message::AssistantContent::Text(text) => {
							text_content = Some(
								text_content
									.map(|mut existing| {
										existing.push('\n');
										existing.push_str(&text.text);
										existing
									})
									.unwrap_or_else(|| text.text.clone()),
							);
						}
						message::AssistantContent::ToolCall(tool_call) => {
							tool_calls.push(tool_call.clone().into());
						}
						message::AssistantContent::Reasoning(_) => {
							return Err(MessageError::ConversionError(
								"Galadriel currently doesn't support reasoning.".into(),
							));
						}
						message::AssistantContent::Image(_) => {
							return Err(MessageError::ConversionError(
								"Galadriel currently doesn't support images.".into(),
							));
						}
					}
				}

				Ok(Self {
					role: "assistant".to_string(),
					content: text_content,
					tool_calls,
				})
			}
		}
	}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
	pub r#type: String,
	pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
	fn from(tool: completion::ToolDefinition) -> Self {
		Self {
			r#type: "function".into(),
			function: tool,
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GaladrielCompletionRequest {
	model: String,
	pub messages: Vec<Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for GaladrielCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		let mut partial_history = vec![];
		if let Some(docs) = req.normalized_documents() {
			partial_history.push(docs);
		}
		partial_history.extend(req.chat_history);

		let mut full_history: Vec<Message> = match &req.preamble {
			Some(preamble) => vec![Message::system(preamble)],
			None => vec![],
		};

		full_history.extend(
			partial_history
				.into_iter()
				.map(message::Message::try_into)
				.collect::<Result<Vec<Message>, _>>()?,
		);

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openai::completion::ToolChoice::try_from)
			.transpose()?;

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			tools: req
				.tools
				.clone()
				.into_iter()
				.map(ToolDefinition::from)
				.collect::<Vec<_>>(),
			tool_choice,
			additional_params: req.additional_params,
		})
	}
}

pub type CompletionModel<T = reqwest::Client> =
	crate::providers::openai_compat::CompletionModel<Galadriel, T>;

impl<T> CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	async fn completion_impl(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "galadriel",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = tracing::field::Empty,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		span.record("gen_ai.system_instructions", &completion_request.preamble);

		let request =
			GaladrielCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Galadriel completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		async move {
			let response = openai_compat::send_and_parse::<
				_,
				openai::CompletionResponse,
				FlatApiError,
				_,
			>(&self.client, req, "Galadriel")
			.await?;

			let span = tracing::Span::current();
			span.record("gen_ai.response.id", response.id.clone());
			span.record("gen_ai.response.model_name", response.model.clone());
			if let Some(ref usage) = response.usage {
				span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
				span.record(
					"gen_ai.usage.output_tokens",
					usage.total_tokens - usage.prompt_tokens,
				);
			}
			response.try_into()
		}
		.instrument(span)
		.await
	}

	async fn stream_impl(
		&self,
		completion_request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
	{
		let preamble = completion_request.preamble.clone();
		let mut request =
			GaladrielCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		let params = json_utils::merge(
			request.additional_params.unwrap_or(serde_json::json!({})),
			serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
		);

		request.additional_params = Some(params);

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat_streaming",
				gen_ai.operation.name = "chat_streaming",
				gen_ai.provider.name = "galadriel",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = preamble,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
				gen_ai.input.messages = serde_json::to_string(&request.messages)?,
				gen_ai.output.messages = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		openai::send_compatible_streaming_request(self.client.clone(), req)
			.instrument(span)
			.await
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = openai::CompletionResponse;
	type StreamingResponse = openai::StreamingCompletionResponse;
	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self {
			client: client.clone(),
			model: model.into(),
		}
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
		self.completion_impl(completion_request).await
	}

	async fn stream(
		&self,
		completion_request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		self.stream_impl(completion_request).await
	}
}
