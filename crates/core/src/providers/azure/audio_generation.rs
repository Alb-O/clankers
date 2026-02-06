use bytes::Bytes;
use serde_json::json;

use super::client::Client;
use crate::audio_generation::{
	self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::HttpClientExt;

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
	client: Client<T>,
	model: String,
}

impl<T> AudioGenerationModel<T> {
	pub fn new(client: Client<T>, deployment_name: impl Into<String>) -> Self {
		Self {
			client,
			model: deployment_name.into(),
		}
	}
}

impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = Bytes;
	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn audio_generation(
		&self,
		request: AudioGenerationRequest,
	) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
		let request = json!({
			"model": self.model,
			"input": request.text,
			"voice": request.voice,
			"speed": request.speed,
		});

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post_audio_generation("/audio/speech")?
			.header("Content-Type", "application/json")
			.body(body)
			.map_err(|e| AudioGenerationError::HttpError(e.into()))?;

		let response = self.client.send::<_, Bytes>(req).await?;
		let status = response.status();
		let response_body = response.into_body().into_future().await?;

		if !status.is_success() {
			return Err(AudioGenerationError::ProviderError(format!(
				"{status}: {}",
				String::from_utf8_lossy(&response_body)
			)));
		}

		Ok(AudioGenerationResponse {
			audio: response_body.to_vec(),
			response: response_body,
		})
	}
}
