use tracing::{Instrument, Level, enabled, info_span};

use super::CompletionsClient as Client;
use super::client::ApiResponse;
use crate::completion;
use crate::completion::{CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::telemetry::SpanCombinator;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub mod streaming;
pub mod types;

use streaming::StreamingCompletionResponse;
use types::*;

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
	pub(crate) client: Client<T>,
	pub model: String,
	pub strict_tools: bool,
	pub tool_result_array_content: bool,
}

impl<T> CompletionModel<T>
where
	T: Default + std::fmt::Debug + Clone + 'static,
{
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
			strict_tools: false,
			tool_result_array_content: false,
		}
	}

	pub fn with_model(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.into(),
			strict_tools: false,
			tool_result_array_content: false,
		}
	}

	/// Enable strict mode for tool schemas.
	///
	/// When enabled, tool schemas are automatically sanitized to meet OpenAI's strict mode requirements:
	/// - `additionalProperties: false` is added to all objects
	/// - All properties are marked as required
	/// - `strict: true` is set on each function definition
	///
	/// This allows OpenAI to guarantee that the model's tool calls will match the schema exactly.
	pub fn with_strict_tools(mut self) -> Self {
		self.strict_tools = true;
		self
	}

	pub fn with_tool_result_array_content(mut self) -> Self {
		self.tool_result_array_content = true;
		self
	}
}

impl CompletionModel<reqwest::Client> {
	pub fn into_agent_builder(self) -> crate::agent::AgentBuilder<Self> {
		crate::agent::AgentBuilder::new(self)
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt
		+ Default
		+ std::fmt::Debug
		+ Clone
		+ WasmCompatSend
		+ WasmCompatSync
		+ 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = super::CompletionsClient<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CoreCompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "openai",
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

		let request = CompletionRequest::try_from(OpenAIRequestParams {
			model: self.model.to_owned(),
			request: completion_request,
			strict_tools: self.strict_tools,
			tool_result_array_content: self.tool_result_array_content,
		})?;

		if enabled!(Level::TRACE) {
			tracing::trace!(
				target: "clankers::completions",
				"OpenAI Chat Completions completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(|e| CompletionError::HttpError(e.into()))?;

		async move {
			let response = self.client.send(req).await?;

			if response.status().is_success() {
				let text = http_client::text(response).await?;

				match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
					ApiResponse::Ok(response) => {
						let span = tracing::Span::current();
						span.record_response_metadata(&response);
						span.record_token_usage(&response.usage);

						if enabled!(Level::TRACE) {
							tracing::trace!(
								target: "clankers::completions",
								"OpenAI Chat Completions completion response: {}",
								serde_json::to_string_pretty(&response)?
							);
						}

						response.try_into()
					}
					ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
				}
			} else {
				let text = http_client::text(response).await?;
				Err(CompletionError::ProviderError(text))
			}
		}
		.instrument(span)
		.await
	}

	async fn stream(
		&self,
		request: CoreCompletionRequest,
	) -> Result<
		crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
		CompletionError,
	> {
		Self::stream(self, request).await
	}
}
