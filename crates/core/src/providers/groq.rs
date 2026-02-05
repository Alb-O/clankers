//! Groq API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::groq;
//!
//! let client = groq::Client::new("YOUR_API_KEY");
//!
//! let gpt4o = client.completion_model(groq::GPT_4O);
//! ```
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Map;

use super::openai::{CompletionResponse, Message as OpenAIMessage, TranscriptionResponse, Usage};
use super::openai_compat::{self, OpenAiCompat, PBuilder};
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm};
use crate::message::{self};
use crate::providers::openai::ToolDefinition;
use crate::transcription::{self, TranscriptionError};

// ================================================================
// Main Groq Client
// ================================================================

#[derive(Debug, Default, Clone, Copy)]
pub struct Groq;

impl OpenAiCompat for Groq {
	const PROVIDER_NAME: &'static str = "groq";
	const BASE_URL: &'static str = "https://api.groq.com/openai/v1";
	const API_KEY_ENV: &'static str = "GROQ_API_KEY";
	const VERIFY_PATH: &'static str = "/models";
	const COMPLETION_PATH: &'static str = "/chat/completions";
	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Capable<TranscriptionModel<H>>;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<Groq, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<PBuilder<Groq>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	/// Create a new Groq client from the `GROQ_API_KEY` environment variable.
	/// Panics if the environment variable is not set.
	fn from_env() -> Self {
		let api_key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set");
		Self::new(&api_key).unwrap()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
	message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
	Ok(T),
	Err(ApiErrorResponse),
}

// ================================================================
// Groq Completion API
// ================================================================

/// The `deepseek-r1-distill-llama-70b` model. Used for chat completion.
pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
/// The `gemma2-9b-it` model. Used for chat completion.
pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
/// The `llama-3.1-8b-instant` model. Used for chat completion.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.2-11b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_11B_VISION_PREVIEW: &str = "llama-3.2-11b-vision-preview";
/// The `llama-3.2-1b-preview` model. Used for chat completion.
pub const LLAMA_3_2_1B_PREVIEW: &str = "llama-3.2-1b-preview";
/// The `llama-3.2-3b-preview` model. Used for chat completion.
pub const LLAMA_3_2_3B_PREVIEW: &str = "llama-3.2-3b-preview";
/// The `llama-3.2-90b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_90B_VISION_PREVIEW: &str = "llama-3.2-90b-vision-preview";
/// The `llama-3.2-70b-specdec` model. Used for chat completion.
pub const LLAMA_3_2_70B_SPECDEC: &str = "llama-3.2-70b-specdec";
/// The `llama-3.2-70b-versatile` model. Used for chat completion.
pub const LLAMA_3_2_70B_VERSATILE: &str = "llama-3.2-70b-versatile";
/// The `llama-guard-3-8b` model. Used for chat completion.
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
/// The `llama3-70b-8192` model. Used for chat completion.
pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
/// The `llama3-8b-8192` model. Used for chat completion.
pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
/// The `mixtral-8x7b-32768` model. Used for chat completion.
pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
	Parsed,
	Raw,
	Hidden,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct GroqCompletionRequest {
	model: String,
	pub messages: Vec<OpenAIMessage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<GroqAdditionalParameters>,
	pub(super) stream: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(super) stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub(super) struct StreamOptions {
	pub(super) include_usage: bool,
}

impl TryFrom<(&str, CompletionRequest)> for GroqCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		// Build up the order of messages (context, chat_history, prompt)
		let mut partial_history = vec![];
		if let Some(docs) = req.normalized_documents() {
			partial_history.push(docs);
		}
		partial_history.extend(req.chat_history);

		// Add preamble to chat history (if available)
		let mut full_history: Vec<OpenAIMessage> = match &req.preamble {
			Some(preamble) => vec![OpenAIMessage::system(preamble)],
			None => vec![],
		};

		// Convert and extend the rest of the history
		full_history.extend(
			partial_history
				.into_iter()
				.map(message::Message::try_into)
				.collect::<Result<Vec<Vec<OpenAIMessage>>, _>>()?
				.into_iter()
				.flatten()
				.collect::<Vec<_>>(),
		);

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openai::ToolChoice::try_from)
			.transpose()?;

		let additional_params: Option<GroqAdditionalParameters> =
			if let Some(params) = req.additional_params {
				Some(serde_json::from_value(params)?)
			} else {
				None
			};

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
			additional_params,
			stream: false,
			stream_options: None,
		})
	}
}

/// Additional parameters to send to the Groq API
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroqAdditionalParameters {
	/// The reasoning format. See Groq's API docs for more details.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_format: Option<ReasoningFormat>,
	/// Whether or not to include reasoning. See Groq's API docs for more details.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub include_reasoning: Option<bool>,
	/// Any other properties not included by default on this struct (that you want to send)
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub extra: Option<Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug)]
pub struct CompletionModel<P, T = reqwest::Client> {
	client: Client<T>,
	/// Name of the model (e.g.: deepseek-r1-distill-llama-70b)
	pub model: String,
	_phantom: std::marker::PhantomData<P>,
}

impl<P, T> CompletionModel<P, T> {
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
			_phantom: std::marker::PhantomData,
		}
	}
}

impl<T> completion::CompletionModel for CompletionModel<Groq, T>
where
	T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = openai_compat::completion_span(
			Groq::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let request = GroqCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "rig::completions",
				"Groq completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;
		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(|e| http_client::Error::Instance(e.into()))?;

		let async_block = async move {
			let response = openai_compat::send_and_parse::<
				_,
				CompletionResponse,
				openai_compat::FlatApiError,
				_,
			>(&self.client, req, "Groq")
			.await?;

			// Record response span manually since groq uses openai::CompletionResponse
			let span = tracing::Span::current();
			openai_compat::record_openai_response_span(&span, &response);

			if tracing::enabled!(tracing::Level::TRACE) {
				tracing::trace!(target: "rig::completions",
					"Groq completion response: {}",
					serde_json::to_string_pretty(&response)?
				);
			}

			response.try_into()
		};

		tracing::Instrument::instrument(async_block, span).await
	}

	async fn stream(
		&self,
		request: CompletionRequest,
	) -> Result<
		crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
		CompletionError,
	> {
		let span =
			openai_compat::streaming_span(Groq::PROVIDER_NAME, &self.model, &request.preamble);

		let mut request = GroqCompletionRequest::try_from((self.model.as_ref(), request))?;

		request.stream = true;
		request.stream_options = Some(StreamOptions {
			include_usage: true,
		});

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "rig::completions",
				"Groq streaming completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;
		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(|e| http_client::Error::Instance(e.into()))?;

		tracing::Instrument::instrument(
			super::openai::send_compatible_streaming_request(self.client.clone(), req),
			span,
		)
		.await
	}
}

// ================================================================
// Groq Transcription API
// ================================================================

pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3_EN: &str = "distil-whisper-large-v3-en";

#[derive(Clone)]
pub struct TranscriptionModel<T> {
	client: Client<T>,
	/// Name of the model (e.g.: gpt-3.5-turbo-1106)
	pub model: String,
}

impl<T> TranscriptionModel<T> {
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}
}
impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
	T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
	type Response = TranscriptionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn transcription(
		&self,
		request: transcription::TranscriptionRequest,
	) -> Result<
		transcription::TranscriptionResponse<Self::Response>,
		transcription::TranscriptionError,
	> {
		let data = request.data;

		let mut body = MultipartForm::new()
			.text("model", self.model.clone())
			.part(Part::bytes("file", data).filename(request.filename.clone()));

		if let Some(language) = request.language {
			body = body.text("language", language);
		}

		if let Some(prompt) = request.prompt {
			body = body.text("prompt", prompt.clone());
		}

		if let Some(ref temperature) = request.temperature {
			body = body.text("temperature", temperature.to_string());
		}

		if let Some(ref additional_params) = request.additional_params {
			for (key, value) in additional_params
				.as_object()
				.expect("Additional Parameters to OpenAI Transcription should be a map")
			{
				body = body.text(key.to_owned(), value.to_string());
			}
		}

		let req = self
			.client
			.post("/audio/transcriptions")?
			.body(body)
			.unwrap();

		let response = self.client.send_multipart::<Bytes>(req).await.unwrap();

		let status = response.status();
		let response_body = response.into_body().into_future().await?.to_vec();

		if status.is_success() {
			match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(&response_body)? {
				ApiResponse::Ok(response) => response.try_into(),
				ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
					api_error_response.message,
				)),
			}
		} else {
			Err(TranscriptionError::ProviderError(
				String::from_utf8_lossy(&response_body).to_string(),
			))
		}
	}
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
	pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();

		usage.input_tokens = self.usage.prompt_tokens as u64;
		usage.total_tokens = self.usage.total_tokens as u64;
		usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
		usage.cached_input_tokens = self
			.usage
			.prompt_tokens_details
			.as_ref()
			.map(|d| d.cached_tokens as u64)
			.unwrap_or(0);

		Some(usage)
	}
}

impl super::openai::CompatStreamingResponse for StreamingCompletionResponse {
	type Usage = Usage;
	fn from_usage(usage: Usage) -> Self {
		Self { usage }
	}
	fn prompt_tokens(usage: &Usage) -> u64 {
		usage.prompt_tokens as u64
	}
	fn output_tokens(usage: &Usage) -> u64 {
		(usage.total_tokens - usage.prompt_tokens) as u64
	}
}

#[cfg(test)]
mod tests {
	use crate::OneOrMany;
	use crate::providers::groq::{GroqAdditionalParameters, GroqCompletionRequest};
	use crate::providers::openai::{Message, UserContent};

	#[test]
	fn serialize_groq_request() {
		let additional_params = GroqAdditionalParameters {
			include_reasoning: Some(true),
			reasoning_format: Some(super::ReasoningFormat::Parsed),
			..Default::default()
		};

		let groq = GroqCompletionRequest {
			model: "openai/gpt-120b-oss".to_string(),
			temperature: None,
			tool_choice: None,
			stream_options: None,
			tools: Vec::new(),
			messages: vec![Message::User {
				content: OneOrMany::one(UserContent::Text {
					text: "Hello world!".to_string(),
				}),
				name: None,
			}],
			stream: false,
			additional_params: Some(additional_params),
		};

		let json = serde_json::to_value(&groq).unwrap();

		assert_eq!(
			json,
			serde_json::json!({
				"model": "openai/gpt-120b-oss",
				"messages": [
					{
						"role": "user",
						"content": [{
							"type": "text",
							"text": "Hello world!"
						}]
					}
				],
				"stream": false,
				"include_reasoning": true,
				"reasoning_format": "parsed"
			})
		)
	}
}
