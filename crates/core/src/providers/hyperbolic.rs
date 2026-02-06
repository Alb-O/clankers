//! Hyperbolic Inference API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::hyperbolic;
//!
//! let client = hyperbolic::Client::new("YOUR_API_KEY");
//!
//! let llama_3_1_8b = client.completion_model(hyperbolic::LLAMA_3_1_8B);
//! ```
use serde::{Deserialize, Serialize};

use super::openai::{AssistantContent, send_compatible_streaming_request};
use crate::OneOrMany;
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;
use crate::providers::openai::Message;
use crate::providers::openai_compat::{
	self, CompletionModel, FlatApiError, OpenAiCompat, PBuilder,
};
use crate::streaming::StreamingCompletionResponse;

#[derive(Debug, Default, Clone, Copy)]
pub struct Hyperbolic;

impl OpenAiCompat for Hyperbolic {
	const PROVIDER_NAME: &'static str = "hyperbolic";
	const BASE_URL: &'static str = "https://api.hyperbolic.xyz";
	const API_KEY_ENV: &'static str = "HYPERBOLIC_API_KEY";
	const VERIFY_PATH: &'static str = "/models";
	const COMPLETION_PATH: &'static str = "/v1/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Capable<ImageGenerationModel<H>>;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Capable<AudioGenerationModel<H>>;
}

pub type Client<H = reqwest::Client> = client::Client<Hyperbolic, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<Hyperbolic>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = BearerAuth;

	/// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
	/// Panics if the environment variable is not set.
	fn from_env() -> Self {
		openai_compat::default_from_env::<Hyperbolic>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(input).unwrap()
	}
}

#[cfg(any(feature = "image", feature = "audio"))]
use crate::providers::openai_compat::ApiResponse;

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
	pub object: String,
	pub embedding: Vec<f64>,
	pub index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
	pub prompt_tokens: usize,
	pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Prompt tokens: {} Total tokens: {}",
			self.prompt_tokens, self.total_tokens
		)
	}
}

/// Meta Llama 3.1b Instruct model with 8B parameters.
pub const LLAMA_3_1_8B: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// Meta Llama 3.3b Instruct model with 70B parameters.
pub const LLAMA_3_3_70B: &str = "meta-llama/Llama-3.3-70B-Instruct";
/// Meta Llama 3.1b Instruct model with 70B parameters.
pub const LLAMA_3_1_70B: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
/// Meta Llama 3 Instruct model with 70B parameters.
pub const LLAMA_3_70B: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
/// Hermes 3 Instruct model with 70B parameters.
pub const HERMES_3_70B: &str = "NousResearch/Hermes-3-Llama-3.1-70b";
/// Deepseek v2.5 model.
pub const DEEPSEEK_2_5: &str = "deepseek-ai/DeepSeek-V2.5";
/// Qwen 2.5 model with 72B parameters.
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct";
/// Meta Llama 3.2b Instruct model with 3B parameters.
pub const LLAMA_3_2_3B: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Qwen 2.5 Coder Instruct model with 32B parameters.
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
/// Preview (latest) version of Qwen model with 32B parameters.
pub const QWEN_QWQ_PREVIEW_32B: &str = "Qwen/QwQ-32B-Preview";
/// Deepseek R1 Zero model.
pub const DEEPSEEK_R1_ZERO: &str = "deepseek-ai/DeepSeek-R1-Zero";
/// Deepseek R1 model.
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";

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
	type StreamingResponse = openai::StreamingCompletionResponse;

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

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
	use base64::Engine;
	use base64::prelude::BASE64_STANDARD;
	use serde::Deserialize;
	use serde_json::json;

	use super::{ApiResponse, Client};
	use crate::http_client::HttpClientExt;
	use crate::image_generation;
	use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
	use crate::json_utils::merge_inplace;

	pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
	pub const SD2: &str = "SD2";
	pub const SD1_5: &str = "SD1.5";
	pub const SSD: &str = "SSD";
	pub const SDXL_TURBO: &str = "SDXL-turbo";
	pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
	pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

	#[derive(Clone)]
	pub struct ImageGenerationModel<T> {
		client: Client<T>,
		pub model: String,
	}

	impl<T> ImageGenerationModel<T> {
		pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
			Self {
				client,
				model: model.into(),
			}
		}

		pub fn with_model(client: Client<T>, model: &str) -> Self {
			Self {
				client,
				model: model.into(),
			}
		}
	}

	#[derive(Clone, Deserialize)]
	pub struct Image {
		image: String,
	}

	#[derive(Clone, Deserialize)]
	pub struct ImageGenerationResponse {
		images: Vec<Image>,
	}

	impl TryFrom<ImageGenerationResponse>
		for image_generation::ImageGenerationResponse<ImageGenerationResponse>
	{
		type Error = ImageGenerationError;

		fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
			let data = BASE64_STANDARD
				.decode(&value.images[0].image)
				.expect("Could not decode image.");

			Ok(Self {
				image: data,
				response: value,
			})
		}
	}

	impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
	where
		T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
	{
		type Response = ImageGenerationResponse;

		type Client = Client<T>;

		fn make(client: &Self::Client, model: impl Into<String>) -> Self {
			Self::new(client.clone(), model)
		}

		async fn image_generation(
			&self,
			generation_request: ImageGenerationRequest,
		) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
		{
			let mut request = json!({
				"model_name": self.model,
				"prompt": generation_request.prompt,
				"height": generation_request.height,
				"width": generation_request.width,
			});

			if let Some(params) = generation_request.additional_params {
				merge_inplace(&mut request, params);
			}

			let body = serde_json::to_vec(&request)?;

			let request = self
				.client
				.post("/v1/image/generation")?
				.header("Content-Type", "application/json")
				.body(body)
				.map_err(|e| ImageGenerationError::HttpError(e.into()))?;

			let response = self.client.send::<_, bytes::Bytes>(request).await?;

			let status = response.status();
			let response_body = response.into_body().into_future().await?.to_vec();

			if !status.is_success() {
				return Err(ImageGenerationError::ProviderError(format!(
					"{status}: {}",
					String::from_utf8_lossy(&response_body)
				)));
			}

			match serde_json::from_slice::<ApiResponse<ImageGenerationResponse>>(&response_body)? {
				ApiResponse::Ok(response) => response.try_into(),
				ApiResponse::Err(err) => Err(ImageGenerationError::ResponseError(err.message)),
			}
		}
	}
}

#[cfg(feature = "audio")]
pub use audio_generation::*;
use tracing::Instrument;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod audio_generation {
	use base64::Engine;
	use base64::prelude::BASE64_STANDARD;
	use bytes::Bytes;
	use serde::Deserialize;
	use serde_json::json;

	use super::{ApiResponse, Client};
	use crate::audio_generation;
	use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
	use crate::http_client::{self, HttpClientExt};

	#[derive(Clone)]
	pub struct AudioGenerationModel<T> {
		client: Client<T>,
		pub language: String,
	}

	#[derive(Clone, Deserialize)]
	pub struct AudioGenerationResponse {
		audio: String,
	}

	impl TryFrom<AudioGenerationResponse>
		for audio_generation::AudioGenerationResponse<AudioGenerationResponse>
	{
		type Error = AudioGenerationError;

		fn try_from(value: AudioGenerationResponse) -> Result<Self, Self::Error> {
			let data = BASE64_STANDARD
				.decode(&value.audio)
				.expect("Could not decode audio.");

			Ok(Self {
				audio: data,
				response: value,
			})
		}
	}

	impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
	where
		T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
	{
		type Response = AudioGenerationResponse;
		type Client = Client<T>;

		fn make(client: &Self::Client, language: impl Into<String>) -> Self {
			Self {
				client: client.clone(),
				language: language.into(),
			}
		}

		async fn audio_generation(
			&self,
			request: AudioGenerationRequest,
		) -> Result<audio_generation::AudioGenerationResponse<Self::Response>, AudioGenerationError>
		{
			let request = json!({
				"language": self.language,
				"speaker": request.voice,
				"text": request.text,
				"speed": request.speed
			});

			let body = serde_json::to_vec(&request)?;

			let req = self
				.client
				.post("/v1/audio/generation")?
				.body(body)
				.map_err(http_client::Error::from)?;

			let response = self.client.send::<_, Bytes>(req).await?;
			let status = response.status();
			let response_body = response.into_body().into_future().await?.to_vec();

			if !status.is_success() {
				return Err(AudioGenerationError::ProviderError(format!(
					"{status}: {}",
					String::from_utf8_lossy(&response_body)
				)));
			}

			match serde_json::from_slice::<ApiResponse<AudioGenerationResponse>>(&response_body)? {
				ApiResponse::Ok(response) => response.try_into(),
				ApiResponse::Err(err) => Err(AudioGenerationError::ProviderError(err.message)),
			}
		}
	}
}
