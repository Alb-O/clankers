use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

use super::client::{ApiResponse, Client};
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
	) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError> {
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
