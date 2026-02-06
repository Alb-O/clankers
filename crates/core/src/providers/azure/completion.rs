use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, Level, enabled, info_span};

use super::client::Client;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils;
use crate::providers::openai;
use crate::providers::openai::completion::streaming::send_compatible_streaming_request;
use crate::providers::openai_compat::ApiResponse;
use crate::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AzureOpenAICompletionRequest {
	model: String,
	pub messages: Vec<openai::completion::types::Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<openai::completion::types::ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openrouter::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for AzureOpenAICompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		//FIXME: Must fix!
		if req.tool_choice.is_some() {
			tracing::warn!(
				"Tool choice is currently not supported in Azure OpenAI. This should be fixed by Clankers 0.25."
			);
		}

		let mut full_history: Vec<openai::completion::types::Message> = match &req.preamble {
			Some(preamble) => vec![openai::completion::types::Message::system(preamble)],
			None => vec![],
		};

		if let Some(docs) = req.normalized_documents() {
			let docs: Vec<openai::completion::types::Message> = docs.try_into()?;
			full_history.extend(docs);
		}

		let chat_history: Vec<openai::completion::types::Message> = req
			.chat_history
			.clone()
			.into_iter()
			.map(|message| message.try_into())
			.collect::<Result<Vec<Vec<openai::completion::types::Message>>, _>>()?
			.into_iter()
			.flatten()
			.collect();

		full_history.extend(chat_history);

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openrouter::ToolChoice::try_from)
			.transpose()?;

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			tools: req
				.tools
				.clone()
				.into_iter()
				.map(openai::completion::types::ToolDefinition::from)
				.collect::<Vec<_>>(),
			tool_choice,
			additional_params: req.additional_params,
		})
	}
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
	client: Client<T>,
	/// Name of the model (e.g.: gpt-4o-mini)
	pub model: String,
}

impl<T> CompletionModel<T> {
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = openai::completion::types::CompletionResponse;
	type StreamingResponse = openai::completion::streaming::StreamingCompletionResponse;
	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model.into())
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<
		completion::CompletionResponse<openai::completion::types::CompletionResponse>,
		CompletionError,
	> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "azure.openai",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = &completion_request.preamble,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		let request =
			AzureOpenAICompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if enabled!(Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Azure OpenAI completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post_chat_completion(&self.model)?
			.body(body)
			.map_err(http_client::Error::from)?;

		async move {
			let response = self.client.send::<_, Bytes>(req).await?;

			let status = response.status();
			let response_body = response.into_body().into_future().await?.to_vec();

			if status.is_success() {
				match serde_json::from_slice::<
					ApiResponse<openai::completion::types::CompletionResponse>,
				>(&response_body)?
				{
					ApiResponse::Ok(response) => {
						let span = tracing::Span::current();
						span.record_response_metadata(&response);
						span.record_token_usage(&response.usage);
						if enabled!(Level::TRACE) {
							tracing::trace!(target: "clankers::completions",
								"Azure OpenAI completion response: {}",
								serde_json::to_string_pretty(&response)?
							);
						}
						response.try_into()
					}
					ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
				}
			} else {
				Err(CompletionError::ProviderError(
					String::from_utf8_lossy(&response_body).to_string(),
				))
			}
		}
		.instrument(span)
		.await
	}

	async fn stream(
		&self,
		completion_request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		let preamble = completion_request.preamble.clone();
		let mut request =
			AzureOpenAICompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		let params = json_utils::merge(
			request.additional_params.unwrap_or(serde_json::json!({})),
			serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
		);

		request.additional_params = Some(params);

		if enabled!(Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Azure OpenAI completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post_chat_completion(&self.model)?
			.body(body)
			.map_err(http_client::Error::from)?;

		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat_streaming",
				gen_ai.operation.name = "chat_streaming",
				gen_ai.provider.name = "azure.openai",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = &preamble,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		tracing_futures::Instrument::instrument(
			send_compatible_streaming_request(self.client.clone(), req),
			span,
		)
		.await
	}
}
