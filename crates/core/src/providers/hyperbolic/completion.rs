use serde::{Deserialize, Serialize};
use tracing::Instrument;

use super::client::{Client, Hyperbolic, Usage};
use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;
use crate::providers::openai::completion::streaming::send_compatible_streaming_request;
use crate::providers::openai::completion::types::{AssistantContent, Message};
use crate::providers::openai_compat::{self, CompletionModel, FlatApiError, OpenAiCompat};
use crate::streaming::StreamingCompletionResponse;

/// A Hyperbolic completion object.
///
/// For more information, see this link: <https://docs.hyperbolic.xyz/reference/create_chat_completion_v1_chat_completions_post>
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
	pub id: String,
	pub object: String,
	pub created: u64,
	pub model: String,
	pub choices: Vec<Choice>,
	pub usage: Option<Usage>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
	type Error = CompletionError;

	fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
		let choice = response.choices.first().ok_or_else(|| {
			CompletionError::ResponseError("Response contained no choices".to_owned())
		})?;

		let content = match &choice.message {
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				let mut content = content
					.iter()
					.map(|c| match c {
						AssistantContent::Text { text } => completion::AssistantContent::text(text),
						AssistantContent::Refusal { refusal } => {
							completion::AssistantContent::text(refusal)
						}
					})
					.collect::<Vec<_>>();

				content.extend(
					tool_calls
						.iter()
						.map(|call| {
							completion::AssistantContent::tool_call(
								&call.id,
								&call.function.name,
								call.function.arguments.clone(),
							)
						})
						.collect::<Vec<_>>(),
				);
				Ok(content)
			}
			_ => Err(CompletionError::ResponseError(
				"Response did not contain a valid message or tool call".into(),
			)),
		}?;

		let choice = OneOrMany::many(content).map_err(|_| {
			CompletionError::ResponseError(
				"Response contained no message or tool call (empty)".to_owned(),
			)
		})?;

		let usage = response
			.usage
			.as_ref()
			.map(|usage| completion::Usage {
				input_tokens: usage.prompt_tokens as u64,
				output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
				total_tokens: usage.total_tokens as u64,
				cached_input_tokens: 0,
			})
			.unwrap_or_default();

		Ok(completion::CompletionResponse {
			choice,
			usage,
			raw_response: response,
		})
	}
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
	pub index: usize,
	pub message: Message,
	pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HyperbolicCompletionRequest {
	model: String,
	pub messages: Vec<Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for HyperbolicCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		if req.tool_choice.is_some() {
			tracing::warn!("WARNING: `tool_choice` not supported on Hyperbolic");
		}

		if !req.tools.is_empty() {
			tracing::warn!("WARNING: `tools` not supported on Hyperbolic");
		}

		let mut full_history: Vec<Message> = match &req.preamble {
			Some(preamble) => vec![Message::system(preamble)],
			None => vec![],
		};

		if let Some(docs) = req.normalized_documents() {
			let docs: Vec<Message> = docs.try_into()?;
			full_history.extend(docs);
		}

		let chat_history: Vec<Message> = req
			.chat_history
			.clone()
			.into_iter()
			.map(|message| message.try_into())
			.collect::<Result<Vec<Vec<Message>>, _>>()?
			.into_iter()
			.flatten()
			.collect();

		full_history.extend(chat_history);

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			additional_params: req.additional_params,
		})
	}
}

impl<T> completion::CompletionModel for CompletionModel<Hyperbolic, T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = openai::completion::streaming::StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = openai_compat::completion_span(
			Hyperbolic::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let request =
			HyperbolicCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Hyperbolic completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/v1/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let async_block = async move {
			let response = openai_compat::send_and_parse::<_, CompletionResponse, FlatApiError, _>(
				&self.client,
				req,
				"Hyperbolic",
			)
			.await?;

			response.try_into()
		};

		tracing::Instrument::instrument(async_block, span).await
	}

	async fn stream(
		&self,
		completion_request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		let span = openai_compat::streaming_span(
			Hyperbolic::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let mut request =
			HyperbolicCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		openai_compat::merge_stream_params(&mut request.additional_params);

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Hyperbolic streaming completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/v1/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		send_compatible_streaming_request(self.client.clone(), req)
			.instrument(span)
			.await
	}
}
