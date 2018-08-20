// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "FileUtils.h"

#include <stdio.h>
#include <fstream>
#include <iostream>

#ifdef USE_LIBPNG
#include <png.h>
#endif

using namespace std;

static const char *pgm_ascii_id = "P2";
static const char *ppm_ascii_id = "P3";
static const char *pgm_id = "P5";
static const char *ppm_id = "P6";
static const char *pfm_id = "Pf";
static bool isPfmFile = false;

typedef enum { MONO_8u, RGB_8u, MONO_16u, MONO_16s, RGBA_8u, FORMAT_UNKNOWN = -1 } FormatType;


struct PNGReaderData {
#ifdef USE_LIBPNG
	png_structp png_ptr;
	png_infop info_ptr;

	PNGReaderData(void)
	{ png_ptr = NULL; info_ptr = NULL; }
	~PNGReaderData(void)
	{ 
		if (info_ptr != NULL) png_destroy_info_struct(png_ptr, &info_ptr);
		if (png_ptr != NULL) png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
	}
#endif
};

static FormatType png_readheader(FILE *fp, int & width, int & height, PNGReaderData & internal)
{
    FormatType type = FORMAT_UNKNOWN;

#ifdef USE_LIBPNG
	png_byte color_type;
	png_byte bit_depth;

	unsigned char header[8];    // 8 is the maximum size that can be checked

	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8)) {
		fprintf(stderr, "Failed to read PNG input: Not actually PNG file\n");
		fprintf(stderr, "Header read: %s\n", header);
        return type;
	}

	/* initialize stuff */
	internal.png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!internal.png_ptr) {
		fprintf(stderr, "png_create_read_struct failed\n");
		return type;
	}

	internal.info_ptr = png_create_info_struct(internal.png_ptr);
	if (!internal.info_ptr) {
		fprintf(stderr, "png_create_info_struct failed\n");
		return type;
	}

	if (setjmp(png_jmpbuf(internal.png_ptr))) {
		fprintf(stderr, "setjmp failed\n");
		return type;
	}

	png_init_io(internal.png_ptr, fp);
	png_set_sig_bytes(internal.png_ptr, 8);

	png_read_info(internal.png_ptr, internal.info_ptr);

	width = png_get_image_width(internal.png_ptr, internal.info_ptr);
	height = png_get_image_height(internal.png_ptr, internal.info_ptr);
	color_type = png_get_color_type(internal.png_ptr, internal.info_ptr);
	bit_depth = png_get_bit_depth(internal.png_ptr, internal.info_ptr);

	if (color_type == PNG_COLOR_TYPE_GRAY) {
		if (bit_depth == 8) type = MONO_8u;
		else if (bit_depth == 16) type = MONO_16u;
		// bit depths 1, 2 and 4 are not accepted
	} else if (color_type == PNG_COLOR_TYPE_RGB) {
		if (bit_depth == 8) type = RGB_8u;
		// bit depth 16 is not accepted
	} else if (color_type == PNG_COLOR_TYPE_RGBA) {
		if (bit_depth == 8) type = RGBA_8u;
		// bit depth 16 is not accepted
	}
	// other color types are not accepted

	if (type == FORMAT_UNKNOWN) {
		fprintf(stderr, "Unknown color type. color_type=%d, bit_depth=%d.\n", color_type, bit_depth);
	}
#endif

	return type;
}

static bool png_readdata(FILE *f, int xsize, int ysize, PNGReaderData & internal, void *data_ext)
{
#ifdef USE_LIBPNG
	if (setjmp(png_jmpbuf(internal.png_ptr))) return false;

	png_read_update_info(internal.png_ptr, internal.info_ptr);

	/* read file */
	if (setjmp(png_jmpbuf(internal.png_ptr))) return false;

	int bytesPerRow = png_get_rowbytes(internal.png_ptr, internal.info_ptr);

	png_byte *data = (png_byte*)data_ext;
	png_bytep *row_pointers = new png_bytep[ysize];
	for (int y=0; y<ysize; y++) row_pointers[y] = &(data[bytesPerRow*y]);

	png_read_image(internal.png_ptr, row_pointers);
	png_read_end(internal.png_ptr, NULL);

	delete[] row_pointers;

	return true;
#else
	return false;
#endif
}

static FormatType pnm_readheader(FILE *f, int *xsize, int *ysize, bool *binary)
{
	char tmp[1024];
	FormatType type = FORMAT_UNKNOWN;
	int xs = 0, ys = 0, max_i = 0;
	bool isBinary = true;

	/* read identifier */
	if (fscanf(f, "%[^ \n\t]", tmp) != 1) {
		fprintf(stderr, "Could not understand pnm identifier.\n");
		return FORMAT_UNKNOWN;
    }
    if (!strcmp(tmp, pgm_id)) type = MONO_8u;                                   //P5  pgm
    else if (!strcmp(tmp, pgm_ascii_id)) { type = MONO_8u; isBinary = false; }  //P2  pgm
    else if (!strcmp(tmp, ppm_id)) type = RGB_8u;                               //P6  ppm
    else if (!strcmp(tmp, ppm_ascii_id)) { type = RGB_8u; isBinary = false; }   //P3  ppm
    else if (!strcmp(tmp, pfm_id)) { type = MONO_16u; isBinary = false; isPfmFile = true; }  //Pf  pfm
	else {
		fprintf(stderr, "Unknown pnm format ID: %s\n", tmp);
		return FORMAT_UNKNOWN;
	}

	/* read size */
	if (!fscanf(f, "%i", &xs)) {
		fprintf(stderr, "Could not read pnm file x-size.\n");
		return FORMAT_UNKNOWN;
	}
	if (!fscanf(f, "%i", &ys)) {
		fprintf(stderr, "Could not read pnm file y-size.\n");
		return FORMAT_UNKNOWN;
    }
    if (strcmp(tmp, pfm_id))
    {
        if (!fscanf(f, "%i", &max_i)) {
            fprintf(stderr, "Could not read pnm file max i.\n");
            return FORMAT_UNKNOWN;
        }
        if (max_i < 0) {
            fprintf(stderr, "Invalid (negative) max_i value in pnm file: %d.\n", max_i);
            return FORMAT_UNKNOWN;
        }
        else if (max_i <= (1 << 8)) {}
        else if ((max_i <= (1 << 15)) && (type == MONO_8u)) type = MONO_16s;
        else if ((max_i <= (1 << 16)) && (type == MONO_8u)) type = MONO_16u;
        else {
            fprintf(stderr, "Invalid  max_i value in pnm file: %d.\n", max_i);
            return FORMAT_UNKNOWN;
        }
    }
	fgetc(f);

	if (xsize) *xsize = xs;
	if (ysize) *ysize = ys;
	if (binary) *binary = isBinary;

	return type;
}

template<class T>
static bool pnm_readdata_ascii_helper(FILE *f, int xsize, int ysize, int channels, T *data)
{
	for (int y = 0; y < ysize; ++y) for (int x = 0; x < xsize; ++x) for (int c = 0; c < channels; ++c) {
		int v;
		if (!fscanf(f, "%i", &v)) return false;
		*data++ = v;
	}
	return true;
}

static bool pnm_readdata_ascii(FILE *f, int xsize, int ysize, FormatType type, void *data)
{
	int channels = 0;
	switch (type)
	{
	case MONO_8u:
		channels = 1;
		return pnm_readdata_ascii_helper(f, xsize, ysize, channels, (unsigned char*)data);
	case RGB_8u:
		channels = 3;
		return pnm_readdata_ascii_helper(f, xsize, ysize, channels, (unsigned char*)data);
	case MONO_16s:
		channels = 1;
		return pnm_readdata_ascii_helper(f, xsize, ysize, channels, (short*)data);
	case MONO_16u:
		channels = 1;
		return pnm_readdata_ascii_helper(f, xsize, ysize, channels, (unsigned short*)data);
	case FORMAT_UNKNOWN:
	default: break;
	}
	return false;
}

static bool pnm_readdata_binary(FILE *f, int xsize, int ysize, FormatType type, void *data)
{
	int channels = 0;
	int bytesPerSample = 0;
	switch (type)
	{
	case MONO_8u: bytesPerSample = sizeof(unsigned char); channels = 1; break;
	case RGB_8u: bytesPerSample = sizeof(unsigned char); channels = 3; break;
	case MONO_16s: bytesPerSample = sizeof(short); channels = 1; break;
	case MONO_16u: bytesPerSample = sizeof(unsigned short); channels = 1; break;
	case FORMAT_UNKNOWN:
	default: break;
	}
	if (bytesPerSample == 0) return false;

	size_t tmp = fread(data, bytesPerSample, xsize*ysize*channels, f);
	if (tmp != (size_t)xsize*ysize*channels) return false;
	return (data != NULL);
}

static bool pnm_writeheader(FILE *f, int xsize, int ysize, FormatType type)
{
	const char *pnmid = NULL;
	int max = 0;
	switch (type) {
	case MONO_8u: pnmid = pgm_id; max = 256; break;
	case RGB_8u: pnmid = ppm_id; max = 255; break;
	case MONO_16s: pnmid = pgm_id; max = 32767; break;
	case MONO_16u: pnmid = pgm_id; max = 65535; break;
	case FORMAT_UNKNOWN:
	default: return false;
	}
	if (pnmid == NULL) return false;

	fprintf(f, "%s\n", pnmid);
	fprintf(f, "%i %i\n", xsize, ysize);
	fprintf(f, "%i\n", max);

	return true;
}

static bool pnm_writedata(FILE *f, int xsize, int ysize, FormatType type, const void *data)
{
	int channels = 0;
	int bytesPerSample = 0;
	switch (type)
	{
	case MONO_8u: bytesPerSample = sizeof(unsigned char); channels = 1; break;
	case RGB_8u: bytesPerSample = sizeof(unsigned char); channels = 3; break;
	case MONO_16s: bytesPerSample = sizeof(short); channels = 1; break;
	case MONO_16u: bytesPerSample = sizeof(unsigned short); channels = 1; break;
	case FORMAT_UNKNOWN:
	default: break;
	}
	fwrite(data, bytesPerSample, channels*xsize*ysize, f);
	return true;
}

void SaveImageToFile(const ITMUChar4Image* image, const char* fileName, bool flipVertical)
{
	FILE *f = fopen(fileName, "wb");
	if (!pnm_writeheader(f, image->noDims.x, image->noDims.y, RGB_8u)) {
		fclose(f); return;
	}

	unsigned char *data = new unsigned char[image->noDims.x*image->noDims.y * 3];

	Vector2i noDims = image->noDims;

	if (flipVertical)
	{
		for (int y = 0; y < noDims.y; y++) for (int x = 0; x < noDims.x; x++)
		{
			int locId_src, locId_dst;
			locId_src = x + y * noDims.x;
			locId_dst = x + (noDims.y - y - 1) * noDims.x;

			data[locId_dst * 3 + 0] = image->GetData(MEMORYDEVICE_CPU)[locId_src].x;
			data[locId_dst * 3 + 1] = image->GetData(MEMORYDEVICE_CPU)[locId_src].y;
			data[locId_dst * 3 + 2] = image->GetData(MEMORYDEVICE_CPU)[locId_src].z;
		}
	}
	else
	{
		for (int i = 0; i < noDims.x * noDims.y; ++i) {
			data[i * 3 + 0] = image->GetData(MEMORYDEVICE_CPU)[i].x;
			data[i * 3 + 1] = image->GetData(MEMORYDEVICE_CPU)[i].y;
			data[i * 3 + 2] = image->GetData(MEMORYDEVICE_CPU)[i].z;
		}
	}

	pnm_writedata(f, image->noDims.x, image->noDims.y, RGB_8u, data);
	delete[] data;
	fclose(f);
}

void SaveImageToFile(const ITMShortImage* image, const char* fileName)
{
	short *data = (short*)malloc(sizeof(short) * image->dataSize);
	const short *dataSource = image->GetData(MEMORYDEVICE_CPU);
	for (size_t i = 0; i < image->dataSize; i++) data[i] = (dataSource[i] << 8) | ((dataSource[i] >> 8) & 255);

	FILE *f = fopen(fileName, "wb");
	if (!pnm_writeheader(f, image->noDims.x, image->noDims.y, MONO_16u)) {
		fclose(f); return;
	}
	pnm_writedata(f, image->noDims.x, image->noDims.y, MONO_16u, data);
	fclose(f);

	delete data;
}

void SaveImageToFile(const ITMFloatImage* image, const char* fileName)
{
	unsigned short *data = new unsigned short[image->dataSize];
	for (size_t i = 0; i < image->dataSize; i++)
	{
		float localData = image->GetData(MEMORYDEVICE_CPU)[i];
		data[i] = localData >= 0 ? (unsigned short)(localData * 1000.0f) : 0;
	}

	FILE *f = fopen(fileName, "wb");
	if (!pnm_writeheader(f, image->noDims.x, image->noDims.y, MONO_16u)) {
		fclose(f); return;
	}
	pnm_writedata(f, image->noDims.x, image->noDims.y, MONO_16u, data);
	fclose(f);

	delete[] data;
}

bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName)
{
  PNGReaderData pngData;
	bool usepng = false;

	int xsize, ysize;
	FormatType type;
	bool binary;
	FILE *f = fopen(fileName, "rb");
  // TODO(andrei): Improve this error handling.

	if (f == NULL) { 
    fprintf(stderr, "Could not open file %s.\n", fileName);
    return false;
  }
    type = pnm_readheader(f, &xsize, &ysize, &binary);

    if ((type != RGB_8u)&&(type != RGBA_8u)) {
		fclose(f);
        f = fopen(fileName, "rb");
		type = png_readheader(f, xsize, ysize, pngData);
		if ((type != RGB_8u)&&(type != RGBA_8u)) {
            fprintf(stderr,"Invalid image type read from PNG header: %d\nAcceptable ones are: %d and %d.\n",type, RGB_8u, RGBA_8u);
			fclose(f);
			return false;
		}
		usepng = true;
	}

	Vector2i newSize(xsize, ysize);
	image->ChangeDims(newSize);
	Vector4u *dataPtr = image->GetData(MEMORYDEVICE_CPU);

	unsigned char *data;
	if (type != RGBA_8u) data = new unsigned char[xsize*ysize * 3];
	else data = (unsigned char*)image->GetData(MEMORYDEVICE_CPU);

  // TODO(andrei): Improve error reporting here.
	if (usepng) {
		if (!png_readdata(f, xsize, ysize, pngData, data)) { fclose(f); delete[] data; return false; }
	} else if (binary) {
		if (!pnm_readdata_binary(f, xsize, ysize, RGB_8u, data)) { fclose(f); delete[] data; return false; }
	} else {
		if (!pnm_readdata_ascii(f, xsize, ysize, RGB_8u, data)) { fclose(f); delete[] data; return false; }
	}
	fclose(f);

	if (type != RGBA_8u)
	{
		for (int i = 0; i < image->noDims.x*image->noDims.y; ++i)
		{
			dataPtr[i].x = data[i * 3 + 0]; dataPtr[i].y = data[i * 3 + 1];
			dataPtr[i].z = data[i * 3 + 2]; dataPtr[i].w = 255;
		}

		delete[] data;
	}

	return true;
}

bool ReadImageFromFile(ITMShortImage *image, const char *fileName)// for depth file to read
{
	PNGReaderData pngData;
	bool usepng = false;

	int xsize, ysize;
	bool binary;
	FILE *f = fopen(fileName, "rb");
	if (f == NULL) {
    fprintf(stderr, "Could not open file %s.\n", fileName);
    return false;
  }
	FormatType type = pnm_readheader(f, &xsize, &ysize, &binary);
    if ((type != MONO_16s) && (type != MONO_16u)) {
		if (type == RGB_8u || type == MONO_8u) {
			fprintf(stderr, "8-bit files are not supported by InfiniTAM. Culprit: [%s]\n", fileName);
			fclose(f);
			return false;
		}
		// Awesome error handling... If the PNM is valid but 8-bit, the system idiotically tries to
		// (uselessly) interpret it as a PNG and starts complaining about the PNG (which was CLEARLY
		// a VALID PNM) is invalid. TODO(andrei): Improve this error handling.
		fclose(f);
		f = fopen(fileName, "rb");    // TODO(andrei): Could we fseek here?
		type = png_readheader(f, xsize, ysize, pngData);
		if ((type != MONO_16s) && (type != MONO_16u)) {
      fprintf(
          stderr,
          "Invalid image type read from PNG header: %d\nAcceptable ones are: %d and %d.\n",
          type, MONO_16s, MONO_16u);
			fclose(f);
			return false;
		}
		usepng = true;
		binary = true;
	}

    cv::Mat pfm_data;
	short *data = new short[xsize*ysize];
    if (usepng) {
            if (!png_readdata(f, xsize, ysize, pngData, data)) {
          fprintf(stderr, "Could not read PNG data.\n");
          fclose(f);
          delete[] data;
          return false;
        }
	} else if (binary) {
            if (!pnm_readdata_binary(f, xsize, ysize, type, data)) {
          fprintf(stderr, "Could not read binary PNM data.\n");
          fclose(f);
          delete[] data;
          return false;
        }
    } else if (isPfmFile)
    {
       fclose(f);
       ReadFilePFM(pfm_data, fileName);////////////  Read PFM disparaty.
       cvMatToShort(pfm_data, data);
    }
    else {
            if (!pnm_readdata_ascii(f, xsize, ysize, type, data)) {
          fprintf(stderr, "Could not read ASCII PNM data.\n");
          fclose(f);
          delete[] data;
          return false;
        }
	}
    if(!isPfmFile) fclose(f);

    Vector2i newSize(xsize, ysize);
    image->ChangeDims(newSize);
	if (binary) {
        for (int i = 0; i < image->noDims.x*image->noDims.y; ++i) {
			image->GetData(MEMORYDEVICE_CPU)[i] = (data[i] << 8) | ((data[i] >> 8) & 255);
		}
    } else {
        for (int i = 0; i < image->noDims.x*image->noDims.y; ++i) {
            image->GetData(MEMORYDEVICE_CPU)[i] = data[i];
		}
	}
	delete[] data;

	return true;
}

void cvMatToShort(cv::Mat pfm_data, short *data) //data in .pfm is disparity
{
    float metersToMillimeters = 1000.0f;
    for(int i=0; i<pfm_data.rows; i++) {
        for(int j=0; j<pfm_data.cols; j++) {
            float disp = pfm_data.at<float>(i,j);
            int32_t depth_mm = static_cast<int32_t>( metersToMillimeters * focal_length_px * baseline_m / disp );
            if (abs(disp) < 1e-5) {
                depth_mm = 0;
            }
            if (depth_mm > 30000 || depth_mm < 500) {
                depth_mm = 0;
            }
            *data = static_cast<int16_t>(depth_mm);
            data ++;
        }
    }
}
