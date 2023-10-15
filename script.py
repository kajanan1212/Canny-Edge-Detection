import cv2
import numpy as np
from contextlib import suppress


def load_image(file_name: str):
    image = cv2.imread(file_name)
    return image


def write_image(file_name: str, image):
    cv2.imwrite(file_name, np.clip(image, 0, 255).astype(np.uint8))


def rgb_to_gray_scale(image):
    RGB = 0.21, 0.72, 0.07
    weights = np.array(RGB).reshape(3, 1)

    gray_scale_image = np.dot(image, weights)

    return np.clip(gray_scale_image, 0, 255).reshape(image.shape[:2]).astype(np.uint8)


def gaussian_kernel(kernel_size, sigma):
    k = kernel_size // 2

    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
        np.exp(-((x - k)**2 + (y - k)**2) / (2 * sigma**2)),
        (kernel_size, kernel_size),
    )

    return kernel


def convolution_2d_with_zero_padding(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    output = np.zeros_like(image, dtype=np.float32)

    flipped_kernel = np.flipud(np.fliplr(kernel))

    for i in range(image_height):
        for j in range(image_width):
            i_start = max(0, i - pad_height)
            i_end = min(image_height, i + pad_height + 1)
            j_start = max(0, j - pad_width)
            j_end = min(image_width, j + pad_width + 1)

            image_region = image[i_start:i_end, j_start:j_end]

            output[i, j] = np.sum(image_region * flipped_kernel[i_start - i + pad_height:i_end - i + pad_height, j_start - j + pad_width:j_end - j + pad_width])

    return output


def remove_noise(image):
    gaussian_kernel_5x5 = gaussian_kernel(kernel_size=5, sigma=1)
    normalized_gaussian_filter_5x5 = gaussian_kernel_5x5 / np.sum(gaussian_kernel_5x5)

    image_after_2d_convolution = convolution_2d_with_zero_padding(
        image=image,
        kernel=normalized_gaussian_filter_5x5,
    )

    return np.clip(image_after_2d_convolution, 0, 255).astype(np.uint8)


def estimate_edge_strength_and_direction(image):
    Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    My = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    Ix = convolution_2d_with_zero_padding(image, Mx)
    Iy = convolution_2d_with_zero_padding(image, My)

    gradient_magnitude = np.hypot(Ix, Iy)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255

    gradient_slope = np.arctan2(Iy, Ix)

    return gradient_magnitude, gradient_slope


def apply_non_maxima_suppression(initial_edge, edge_direction):
    ie_height, ie_width = initial_edge.shape
    one_px_thick_edge = np.zeros((ie_height, ie_width), dtype=np.float32)

    angle = edge_direction * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, ie_height - 1):
        for j in range(1, ie_width - 1):
            with suppress(IndexError):
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = initial_edge[i, j+1]
                    r = initial_edge[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = initial_edge[i+1, j-1]
                    r = initial_edge[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = initial_edge[i+1, j]
                    r = initial_edge[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = initial_edge[i-1, j-1]
                    r = initial_edge[i+1, j+1]

                if (initial_edge[i, j] >= q) and (initial_edge[i, j] >= r):
                    one_px_thick_edge[i, j] = initial_edge[i, j]
                else:
                    one_px_thick_edge[i, j] = 0
    
    return one_px_thick_edge


def apply_double_threshold(
    one_px_thick_edge, low_threshold, high_threshold
):
    canny_edge = np.zeros(one_px_thick_edge.shape, dtype=np.uint8)

    strong_i, strong_j = np.where(one_px_thick_edge >= high_threshold)

    weak_i, weak_j = np.where(
        (low_threshold < one_px_thick_edge) & (one_px_thick_edge < high_threshold)
    )

    canny_edge[strong_i, strong_j] = 255
    canny_edge[weak_i, weak_j] = 15

    return np.clip(canny_edge, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # 1.Load an image from the disk
    image = load_image(file_name="images/01-image.jpg")
    print(f"original image's shape: {image.shape}")

    # 2.Convert the image to gray-scale (8bpp format)
    gray_scale = rgb_to_gray_scale(image=image)
    print(f"gray scale image's shape: {gray_scale.shape}")
    write_image("images/02-gray-scale.jpg", gray_scale)

    # 3.Filter image with a Gaussian kernel to remove noise
    noise_removed = remove_noise(image=gray_scale)
    print(f"noise removed image's shape: {noise_removed.shape}")
    write_image("images/03-noise-removed.jpg", noise_removed)

    # 4.Estimate gradient strength and direction
    initial_edge, edge_direction = (
        estimate_edge_strength_and_direction(
            image=noise_removed,
        )
    )
    print(f"initial edge image's shape: {initial_edge.shape}")
    write_image("images/04-initial-edge.jpg", initial_edge)

    # 5.Ensure the edge is one pixel thick by suppressing
    # non maximum gradients along direction normal to the gradient
    one_px_thick_edge = apply_non_maxima_suppression(
        initial_edge=initial_edge,
        edge_direction=edge_direction,
    )
    print(f"one pixel thick edge image's shape: {one_px_thick_edge.shape}")
    write_image("images/05-one-px-thick-edge.jpg", one_px_thick_edge)

    # 6.Link edge maximum gradient pixels using a dual threshold
    canny_edge = apply_double_threshold(
        one_px_thick_edge=one_px_thick_edge,
        low_threshold=1,
        high_threshold=12,
    )
    print(f"canny edge image's shape: {canny_edge.shape}")
    write_image("images/06-canny-edge.jpg", canny_edge)
