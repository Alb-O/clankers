//! The OpenAI Responses API.
//!
//! By default when creating a completion client, this is the API that gets used.
//!
//! If you'd like to switch back to the regular Completions API, you can do so by using the `.completions_api()` function - see below for an example:
//! ```rust
//! let openai_client = clankers::providers::openai::Client::from_env();
//! let model = openai_client.completion_model("gpt-4o").completions_api();
//! ```
use tracing::{Instrument, Level, enabled, info_span};

use super::Client;
use super::responses_api::streaming::StreamingCompletionResponse;
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{completion, http_client};

pub mod streaming;
pub mod types;
pub use types::*;

/// The completion model struct for OpenAI's response API.
#[derive(Clone)]
pub struct ResponsesCompletionModel<T = reqwest::Client> {
	/// The OpenAI client
	pub(crate) client: Client<T>,
	/// Name of the model (e.g.: gpt-3.5-turbo-1106)
	pub model: String,
}

impl<T> ResponsesCompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
	/// Creates a new [`ResponsesCompletionModel`].
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}

	pub fn with_model(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.to_string(),
		}
	}

	/// Use the Completions API instead of Responses.
	pub fn completions_api(self) -> crate::providers::openai::completion::CompletionModel<T> {
		super::completion::CompletionModel::with_model(self.client.completions_api(), &self.model)
	}

	/// Attempt to create a completion request from [`crate::completion::CompletionRequest`].
	pub(crate) fn create_completion_request(
		&self,
		completion_request: crate::completion::CompletionRequest,
	) -> Result<CompletionRequest, CompletionError> {
		let req = CompletionRequest::try_from((self.model.clone(), completion_request))?;

		Ok(req)
	}
}

impl<T> completion::CompletionModel for ResponsesCompletionModel<T>
where
	T: HttpClientExt
		+ Clone
		+ std::fmt::Debug
		+ Default
		+ WasmCompatSend
		+ WasmCompatSync
		+ 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = super::Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: crate::completion::CompletionRequest,
	) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = tracing::field::Empty,
				gen_ai.request.model = tracing::field::Empty,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
				gen_ai.input.messages = tracing::field::Empty,
				gen_ai.output.messages = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		span.record("gen_ai.provider.name", "openai");
		span.record("gen_ai.request.model", &self.model);
		let request = self.create_completion_request(completion_request)?;
		let body = serde_json::to_vec(&request)?;

		if enabled!(Level::TRACE) {
			tracing::trace!(
				target: "clankers::completions",
				"OpenAI Responses completion request: {request}",
				request = serde_json::to_string_pretty(&request)?
			);
		}

		let req = self
			.client
			.post("/responses")?
			.body(body)
			.map_err(|e| CompletionError::HttpError(e.into()))?;

		async move {
			let response = self.client.send(req).await?;

			if response.status().is_success() {
				let t = http_client::text(response).await?;
				let response = serde_json::from_str::<Self::Response>(&t)?;
				let span = tracing::Span::current();
				span.record("gen_ai.response.id", &response.id);
				span.record("gen_ai.response.model", &response.model);
				if let Some(ref usage) = response.usage {
					span.record("gen_ai.usage.output_tokens", usage.output_tokens);
					span.record("gen_ai.usage.input_tokens", usage.input_tokens);
				}
				if enabled!(Level::TRACE) {
					tracing::trace!(
						target: "clankers::completions",
						"OpenAI Responses completion response: {response}",
						response = serde_json::to_string_pretty(&response)?
					);
				}
				response.try_into()
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
		request: crate::completion::CompletionRequest,
	) -> Result<
		crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
		CompletionError,
	> {
		ResponsesCompletionModel::stream(self, request).await
	}
}
