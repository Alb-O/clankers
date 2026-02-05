//! Shared infrastructure for OpenAI-compatible providers.
//!
//! This module extracts the common boilerplate that all OpenAI-compatible providers
//! (moonshot, hyperbolic, perplexity, mira, groq, deepseek, galadriel) share into
//! a single trait + blanket implementation set.
//!
//! # Usage
//! ```ignore
//! use crate::providers::openai_compat::{OpenAiCompat, PBuilder};
//!
//! #[derive(Debug, Default, Clone, Copy)]
//! pub struct MyProvider;
//!
//! impl OpenAiCompat for MyProvider {
//!     const PROVIDER_NAME: &'static str = "my_provider";
//!     const BASE_URL: &'static str = "https://api.example.com/v1";
//!     const API_KEY_ENV: &'static str = "MY_PROVIDER_API_KEY";
//!     const VERIFY_PATH: &'static str = "/models";
//!     const COMPLETION_PATH: &'static str = "/chat/completions";
//!     // ...
//! }
//! ```

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{Instrument, info_span};

use crate::client::{self, BearerAuth, Capabilities, DebugExt, Provider, ProviderBuilder};
use crate::completion::CompletionError;
use crate::http_client::{self, HttpClientExt};
use crate::json_utils;
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;

// ================================================================
// OpenAiCompat trait
// ================================================================

/// Core trait for OpenAI-compatible providers. Implementing this gives you blanket
/// impls of `Provider`, `ProviderBuilder`, `DebugExt`, and `Capabilities`.
pub trait OpenAiCompat: Debug + Clone + Default + Send + Sync + Sized + 'static {
	const PROVIDER_NAME: &'static str;
	const BASE_URL: &'static str;
	const API_KEY_ENV: &'static str;
	const VERIFY_PATH: &'static str;
	const COMPLETION_PATH: &'static str;

	/// Extra builder-side state (e.g. Galadriel's `fine_tune_api_key`). Defaults to `()`.
	type BuilderState: Debug + Clone + Default + Send + Sync;

	// Capability GATs — providers set these to `Capable<Model<H>>` or `Nothing`.
	type Completion<H>;
	type Embeddings<H>;
	type Transcription<H>;
	#[cfg(feature = "image")]
	type ImageGeneration<H>;
	#[cfg(feature = "audio")]
	type AudioGeneration<H>;

	/// Override to read builder state during `Provider::build` (e.g. Galadriel).
	fn build_from<H>(
		_builder: &client::ClientBuilder<PBuilder<Self>, BearerAuth, H>,
	) -> http_client::Result<Self> {
		Ok(Self::default())
	}

	/// Override to expose ext fields in Debug output (e.g. Galadriel).
	fn debug_fields(&self) -> Vec<(&'static str, &dyn Debug)> {
		vec![]
	}
}

// ================================================================
// PBuilder<P> — generic provider builder
// ================================================================

#[derive(Debug, Clone)]
pub struct PBuilder<P: OpenAiCompat> {
	pub state: P::BuilderState,
}

impl<P: OpenAiCompat> Default for PBuilder<P> {
	fn default() -> Self {
		Self {
			state: P::BuilderState::default(),
		}
	}
}

// ================================================================
// Blanket: Provider for P
// ================================================================

impl<P: OpenAiCompat> Provider for P {
	type Builder = PBuilder<P>;

	const VERIFY_PATH: &'static str = P::VERIFY_PATH;

	fn build<H>(
		builder: &client::ClientBuilder<
			Self::Builder,
			<Self::Builder as ProviderBuilder>::ApiKey,
			H,
		>,
	) -> http_client::Result<Self> {
		P::build_from(builder)
	}
}

// ================================================================
// Blanket: ProviderBuilder for PBuilder<P>
// ================================================================

impl<P: OpenAiCompat> ProviderBuilder for PBuilder<P> {
	type Output = P;
	type ApiKey = BearerAuth;

	const BASE_URL: &'static str = P::BASE_URL;
}

// ================================================================
// Blanket: DebugExt for P
// ================================================================

impl<P: OpenAiCompat> DebugExt for P {
	fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
		self.debug_fields().into_iter()
	}
}

// ================================================================
// Blanket: Capabilities<H> for P
// ================================================================

impl<P, H> Capabilities<H> for P
where
	P: OpenAiCompat<
			Completion<H>: crate::client::Capability,
			Embeddings<H>: crate::client::Capability,
			Transcription<H>: crate::client::Capability,
		>,
	#[cfg(feature = "image")]
	P: OpenAiCompat<ImageGeneration<H>: crate::client::Capability>,
	#[cfg(feature = "audio")]
	P: OpenAiCompat<AudioGeneration<H>: crate::client::Capability>,
{
	type Completion = P::Completion<H>;
	type Embeddings = P::Embeddings<H>;
	type Transcription = P::Transcription<H>;
	#[cfg(feature = "image")]
	type ImageGeneration = P::ImageGeneration<H>;
	#[cfg(feature = "audio")]
	type AudioGeneration = P::AudioGeneration<H>;
}

// ================================================================
// Generic CompletionModel<P, T>
// ================================================================

#[derive(Clone)]
pub struct CompletionModel<P: OpenAiCompat, T = reqwest::Client> {
	pub(crate) client: client::Client<P, T>,
	pub model: String,
}

impl<P: OpenAiCompat, T> CompletionModel<P, T> {
	pub fn new(client: client::Client<P, T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}

	pub fn with_model(client: client::Client<P, T>, model: &str) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}
}

// ================================================================
// Shared error types
// ================================================================

/// A flat API error response: `{ "message": "..." }`
#[derive(Debug, Deserialize)]
pub struct FlatApiError {
	pub message: String,
}

/// A nested API error response: `{ "error": { "message": "..." } }`
#[derive(Debug, Deserialize)]
pub struct NestedApiError {
	pub error: FlatApiError,
}

/// Unified API response enum for OpenAI-compatible providers.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T, E = FlatApiError> {
	Ok(T),
	Err(E),
}

impl From<FlatApiError> for CompletionError {
	fn from(err: FlatApiError) -> Self {
		CompletionError::ProviderError(err.message)
	}
}

impl From<NestedApiError> for CompletionError {
	fn from(err: NestedApiError) -> Self {
		CompletionError::ProviderError(err.error.message)
	}
}

// ================================================================
// Tracing span helpers
// ================================================================

pub fn completion_span(provider: &str, model: &str, preamble: &Option<String>) -> tracing::Span {
	let span = if tracing::Span::current().is_disabled() {
		info_span!(
			target: "rig::completions",
			"chat",
			gen_ai.operation.name = "chat",
			gen_ai.provider.name = %provider,
			gen_ai.request.model = model,
			gen_ai.system_instructions = tracing::field::Empty,
			gen_ai.response.id = tracing::field::Empty,
			gen_ai.response.model = tracing::field::Empty,
			gen_ai.usage.output_tokens = tracing::field::Empty,
			gen_ai.usage.input_tokens = tracing::field::Empty,
		)
	} else {
		tracing::Span::current()
	};
	span.record("gen_ai.system_instructions", preamble);
	span
}

pub fn streaming_span(provider: &str, model: &str, preamble: &Option<String>) -> tracing::Span {
	let span = if tracing::Span::current().is_disabled() {
		info_span!(
			target: "rig::completions",
			"chat_streaming",
			gen_ai.operation.name = "chat_streaming",
			gen_ai.provider.name = %provider,
			gen_ai.request.model = model,
			gen_ai.system_instructions = tracing::field::Empty,
			gen_ai.response.id = tracing::field::Empty,
			gen_ai.response.model = tracing::field::Empty,
			gen_ai.usage.output_tokens = tracing::field::Empty,
			gen_ai.usage.input_tokens = tracing::field::Empty,
		)
	} else {
		tracing::Span::current()
	};
	span.record("gen_ai.system_instructions", preamble);
	span
}

/// Record common OpenAI-style response fields onto the current span.
pub fn record_openai_response_span(span: &tracing::Span, response: &openai::CompletionResponse) {
	span.record("gen_ai.response.id", response.id.clone());
	span.record("gen_ai.response.model_name", response.model.clone());
	if let Some(ref usage) = response.usage {
		span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
		span.record(
			"gen_ai.usage.output_tokens",
			usage.total_tokens - usage.prompt_tokens,
		);
	}
}

// ================================================================
// HTTP send + parse helper
// ================================================================

/// Send an HTTP request, parse the response as `ApiResponse<Resp, Err>`, and return the
/// successful variant or a `CompletionError`.
pub async fn send_and_parse<P, Resp, Err, T>(
	client: &client::Client<P, T>,
	req: http::Request<Vec<u8>>,
	provider_name: &str,
) -> Result<Resp, CompletionError>
where
	P: Provider + Send + Sync + 'static,
	T: HttpClientExt + Clone + Send + 'static,
	Resp: serde::de::DeserializeOwned + Debug + Serialize,
	Err: serde::de::DeserializeOwned + Debug + Into<CompletionError>,
{
	let response = client.send::<_, bytes::Bytes>(req).await?;

	let status = response.status();
	let response_body = response.into_body().into_future().await?.to_vec();

	if status.is_success() {
		match serde_json::from_slice::<ApiResponse<Resp, Err>>(&response_body)? {
			ApiResponse::Ok(resp) => {
				if tracing::enabled!(tracing::Level::TRACE) {
					tracing::trace!(
						target: "rig::completions",
						"{} completion response: {}",
						provider_name,
						serde_json::to_string_pretty(&resp)?
					);
				}
				Ok(resp)
			}
			ApiResponse::Err(err) => Err(err.into()),
		}
	} else {
		Err(CompletionError::ProviderError(
			String::from_utf8_lossy(&response_body).to_string(),
		))
	}
}

// ================================================================
// Streaming helpers
// ================================================================

/// Merge `stream: true` and `stream_options: { include_usage: true }` into
/// `additional_params`, creating the object if it's `None`.
pub fn merge_stream_params(additional_params: &mut Option<Value>) {
	let params = json_utils::merge(
		additional_params.take().unwrap_or(serde_json::json!({})),
		serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
	);
	*additional_params = Some(params);
}

/// Standard streaming flow: merge stream params, serialize, post, delegate to
/// `openai::send_compatible_streaming_request`.
pub async fn stream_with_openai_compat<P, T>(
	client: &client::Client<P, T>,
	body: Vec<u8>,
	span: tracing::Span,
) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
where
	P: OpenAiCompat,
	T: HttpClientExt + Clone + Default + Debug + Send + 'static,
{
	let req = client
		.post(P::COMPLETION_PATH)?
		.body(body)
		.map_err(http_client::Error::from)?;

	send_compatible_streaming_request(client.clone(), req)
		.instrument(span)
		.await
}

// ================================================================
// ProviderClient helper
// ================================================================

/// Default `from_env()` implementation: reads `P::API_KEY_ENV` and builds a client.
pub fn default_from_env<P>() -> client::Client<P>
where
	P: OpenAiCompat,
{
	let api_key =
		std::env::var(P::API_KEY_ENV).unwrap_or_else(|_| panic!("{} not set", P::API_KEY_ENV));
	client::Client::new(&api_key).unwrap()
}
