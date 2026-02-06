use bytes::Bytes;
use serde_json::json;

use super::client::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::providers::openai::ImageGenerationResponse;
use crate::providers::openai_compat::ApiResponse;

#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
	client: Client<T>,
	pub model: String,
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = ImageGenerationResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self {
			client: client.clone(),
			model: model.into(),
		}
	}

	async fn image_generation(
		&self,
		generation_request: ImageGenerationRequest,
	) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError> {
		let request = json!({
			"model": self.model,
			"prompt": generation_request.prompt,
			"size": format!("{}x{}", generation_request.width, generation_request.height),
			"response_format": "b64_json"
		});

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post_image_generation(&self.model)?
			.body(body)
			.map_err(|e| ImageGenerationError::HttpError(e.into()))?;

		let response = self.client.send::<_, Bytes>(req).await?;
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
			ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
		}
	}
}
