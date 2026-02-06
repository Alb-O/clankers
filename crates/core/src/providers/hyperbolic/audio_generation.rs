use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;

use super::client::{ApiResponse, Client};
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
	) -> Result<audio_generation::AudioGenerationResponse<Self::Response>, AudioGenerationError> {
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
