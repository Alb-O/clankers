//! Moonshot API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::moonshot;
//!
//! let client = moonshot::Client::new("YOUR_API_KEY");
//!
//! let moonshot_model = client.completion_model(moonshot::MOONSHOT_CHAT);
//! ```
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::providers::openai;
use crate::providers::openai::completion::streaming::send_compatible_streaming_request;
use crate::providers::openai_compat::{self, FlatApiError, OpenAiCompat, PBuilder};
use crate::streaming::StreamingCompletionResponse;
use crate::{http_client, message};

#[derive(Debug, Default, Clone, Copy)]
pub struct Moonshot;

impl OpenAiCompat for Moonshot {
	const PROVIDER_NAME: &'static str = "moonshot";
	const BASE_URL: &'static str = "https://api.moonshot.cn/v1";
	const API_KEY_ENV: &'static str = "MOONSHOT_API_KEY";
	const VERIFY_PATH: &'static str = "/models";
	const COMPLETION_PATH: &'static str = "/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<openai_compat::CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<Moonshot, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<Moonshot>, BearerAuth, H>;
pub type CompletionModel<T = reqwest::Client> = openai_compat::CompletionModel<Moonshot, T>;

impl ProviderClient for Client {
	type Input = String;

	/// Create a new Moonshot client from the `MOONSHOT_API_KEY` environment variable.
	/// Panics if the environment variable is not set.
	fn from_env() -> Self {
		openai_compat::default_from_env::<Moonshot>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MoonshotCompletionRequest {
	model: String,
	pub messages: Vec<openai::completion::types::Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<openai::completion::types::ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openai::completion::types::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for MoonshotCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		let mut partial_history = vec![];
		if let Some(docs) = req.normalized_documents() {
			partial_history.push(docs);
		}
		partial_history.extend(req.chat_history);

		let mut full_history: Vec<openai::completion::types::Message> = match &req.preamble {
			Some(preamble) => vec![openai::completion::types::Message::system(preamble)],
			None => vec![],
		};

		full_history.extend(
			partial_history
				.into_iter()
				.map(message::Message::try_into)
				.collect::<Result<Vec<Vec<openai::completion::types::Message>>, _>>()?
				.into_iter()
				.flatten()
				.collect::<Vec<_>>(),
		);

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openai::completion::types::ToolChoice::try_from)
			.transpose()?;

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			max_tokens: req.max_tokens,
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

impl<T> completion::CompletionModel for openai_compat::CompletionModel<Moonshot, T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = openai::completion::types::CompletionResponse;
	type StreamingResponse = openai::completion::streaming::StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<
		completion::CompletionResponse<openai::completion::types::CompletionResponse>,
		CompletionError,
	> {
		let span = openai_compat::completion_span(
			Moonshot::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let request =
			MoonshotCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"MoonShot completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;
		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let async_block = async move {
			let response = openai_compat::send_and_parse::<
				_,
				openai::completion::types::CompletionResponse,
				FlatApiError,
				_,
			>(&self.client, req, "MoonShot")
			.await?;

			let span = tracing::Span::current();
			openai_compat::record_openai_response_span(&span, &response);
			response.try_into()
		};

		async_block.instrument(span).await
	}

	async fn stream(
		&self,
		request: CompletionRequest,
	) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		let span =
			openai_compat::streaming_span(Moonshot::PROVIDER_NAME, &self.model, &request.preamble);

		let mut request = MoonshotCompletionRequest::try_from((self.model.as_ref(), request))?;

		openai_compat::merge_stream_params(&mut request.additional_params);

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"MoonShot streaming completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;
		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		send_compatible_streaming_request(self.client.clone(), req)
			.instrument(span)
			.await
	}
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub enum ToolChoice {
	None,
	#[default]
	Auto,
}

impl TryFrom<message::ToolChoice> for ToolChoice {
	type Error = CompletionError;

	fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
		let res = match value {
			message::ToolChoice::None => Self::None,
			message::ToolChoice::Auto => Self::Auto,
			choice => {
				return Err(CompletionError::ProviderError(format!(
					"Unsupported tool choice type: {choice:?}"
				)));
			}
		};

		Ok(res)
	}
}
