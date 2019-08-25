#pragma once

#include <string>
#include <optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

using  namespace optix;

//get ptx Path
static std::string ptxPath(const std::string& cuda_file)
{
	return std::string("./ptx/" + cuda_file + ".ptx");
}


void SavePPM(const unsigned char *Pix, const char *fname, int wid, int hgt, int chan)
{
	if (Pix == NULL || wid < 1 || hgt < 1)
		throw Exception("Image is ill-formed. Not saving");

	if (chan != 1 && chan != 3 && chan != 4)
		throw Exception("Attempting to save image with channel count != 1, 3, or 4.");

	std::ofstream OutFile(fname, std::ios::out | std::ios::binary);
	if (!OutFile.is_open())
		throw Exception("Could not open file for SavePPM");

	bool is_float = false;
	OutFile << 'P';
	OutFile << ((chan == 1 ? (is_float ? 'Z' : '5') : (chan == 3 ? (is_float ? '7' : '6') : '8'))) << std::endl;
	OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

	OutFile.write(reinterpret_cast<char*>(const_cast<unsigned char*>(Pix)), wid * hgt * chan * (is_float ? 4 : 1));

	OutFile.close();
}

void displayBufferPPM(const char* filename, RTbuffer buffer, bool disable_srgb_conversion)
{
	GLsizei width, height;
	RTsize buffer_width, buffer_height;

	GLvoid* imageData;
	RT_CHECK_ERROR(rtBufferMap(buffer, &imageData));

	RT_CHECK_ERROR(rtBufferGetSize2D(buffer, &buffer_width, &buffer_height));
	width = static_cast<GLsizei>(buffer_width);
	height = static_cast<GLsizei>(buffer_height);

	std::vector<unsigned char> pix(width * height * 3);

	RTformat buffer_format;
	RT_CHECK_ERROR(rtBufferGetFormat(buffer, &buffer_format));

	const float gamma_inv = 1.0f / 2.2f;

	switch (buffer_format) {
	case RT_FORMAT_UNSIGNED_BYTE4:
		// Data is BGRA and upside down, so we need to swizzle to RGB
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + (3 * width*(height - 1 - j));
			unsigned char *src = ((unsigned char*)imageData) + (4 * width*j);
			for (int i = 0; i < width; i++) {
				*dst++ = *(src + 2);
				*dst++ = *(src + 1);
				*dst++ = *(src + 0);
				src += 4;
			}
		}
		break;

	case RT_FORMAT_FLOAT:
		// This buffer is upside down
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + width*(height - 1 - j);
			float* src = ((float*)imageData) + (3 * width*j);
			for (int i = 0; i < width; i++) {
				int P;
				if (disable_srgb_conversion)
					P = static_cast<int>((*src++) * 255.0f);
				else
					P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
				unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;

				// write the pixel to all 3 channels
				*dst++ = static_cast<unsigned char>(Clamped);
				*dst++ = static_cast<unsigned char>(Clamped);
				*dst++ = static_cast<unsigned char>(Clamped);
			}
		}
		break;

	case RT_FORMAT_FLOAT3:
		// This buffer is upside down
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + (3 * width*(height - 1 - j));
			float* src = ((float*)imageData) + (3 * width*j);
			for (int i = 0; i < width; i++) {
				for (int elem = 0; elem < 3; ++elem) {
					int P;
					if (disable_srgb_conversion)
						P = static_cast<int>((*src++) * 255.0f);
					else
						P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
					unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
					*dst++ = static_cast<unsigned char>(Clamped);
				}
			}
		}
		break;

	case RT_FORMAT_FLOAT4:
		// This buffer is upside down
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + (3 * width*(height - 1 - j));
			float* src = ((float*)imageData) + (4 * width*j);
			for (int i = 0; i < width; i++) {
				for (int elem = 0; elem < 3; ++elem) {
					int P;
					if (disable_srgb_conversion)
						P = static_cast<int>((*src++) * 255.0f);
					else
						P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
					unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
					*dst++ = static_cast<unsigned char>(Clamped);
				}
				// skip alpha
				src++;
			}
		}
		break;

	default:
		fprintf(stderr, "Unrecognized buffer data type or format.\n");
		exit(2);
		break;
	}

	SavePPM(&pix[0], filename, width, height, 3);

	// Now unmap the buffer
	RT_CHECK_ERROR(rtBufferUnmap(buffer));
}