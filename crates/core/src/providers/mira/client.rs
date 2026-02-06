use std::string::FromUtf8Error;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::completion::CompletionModel;
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai_compat::{self, OpenAiCompat};

#[derive(Debug, Default, Clone, Copy)]
pub struct Mira;

impl OpenAiCompat for Mira {
	const PROVIDER_NAME: &'static str = "mira";
	const BASE_URL: &'static str = "https://api.mira.network";
	const API_KEY_ENV: &'static str = "MIRA_API_KEY";
	const VERIFY_PATH: &'static str = "/user-credits";
	const COMPLETION_PATH: &'static str = "/v1/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;

	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;

	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<Mira, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<openai_compat::PBuilder<Mira>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	fn from_env() -> Self {
		openai_compat::default_from_env::<Mira>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

#[derive(Debug, Error)]
pub enum MiraError {
	#[error("Invalid API key")]
	InvalidApiKey,
	#[error("API error: {0}")]
	ApiError(u16),
	#[error("Request error: {0}")]
	RequestError(#[from] http_client::Error),
	#[error("UTF-8 error: {0}")]
	Utf8Error(#[from] FromUtf8Error),
	#[error("JSON error: {0}")]
	JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelsResponse {
	data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
	id: String,
}

impl<T> Client<T>
where
	T: HttpClientExt + 'static,
{
	/// List available models
	pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
		let req = self.get("/v1/models").and_then(|req| {
			req.body(http_client::NoBody)
				.map_err(http_client::Error::Protocol)
		})?;

		let response = self.send(req).await?;

		let status = response.status();

		if !status.is_success() {
			// Log the error text but don't store it in an unused variable
			let error_text = http_client::text(response).await.unwrap_or_default();
			tracing::error!("Error response: {}", error_text);
			return Err(MiraError::ApiError(status.as_u16()));
		}

		let response_text = http_client::text(response).await?;

		let models: ModelsResponse = serde_json::from_str(&response_text).map_err(|e| {
			tracing::error!("Failed to parse response: {}", e);
			MiraError::JsonError(e)
		})?;

		Ok(models.data.into_iter().map(|model| model.id).collect())
	}
}
