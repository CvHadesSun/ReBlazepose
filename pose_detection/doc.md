# Rect and NormalizeRect
```// A rectangle with rotation in image coordinates.
message Rect {
  // Location of the center of the rectangle in image coordinates.
  // The (0, 0) point is at the (top, left) corner.
  required int32 x_center = 1;
  required int32 y_center = 2;

  // Size of the rectangle.
  required int32 height = 3;
  required int32 width = 4;

  // Rotation angle is clockwise in radians.
  optional float rotation = 5 [default = 0.0];

  // Optional unique id to help associate different Rects to each other.
  optional int64 rect_id = 6;
}

// A rectangle with rotation in normalized coordinates. The values of box center
// location and size are within [0, 1].
message NormalizedRect {
  // Location of the center of the rectangle in image coordinates.
  // The (0.0, 0.0) point is at the (top, left) corner.
  required float x_center = 1;
  required float y_center = 2;

  // Size of the rectangle.
  required float height = 3;
  required float width = 4;

  // Rotation angle is clockwise in radians.
  optional float rotation = 5 [default = 0.0];

  // Optional unique id to help associate different NormalizedRects to each
  // other.
  optional int64 rect_id = 6;
}
```

#image frame format
```
message ImageFormat {
  enum Format {
    // The format is unknown.  It is not valid for an ImageFrame to be
    // initialized with this value.
    UNKNOWN = 0;

    // sRGB, interleaved: one byte for R, then one byte for G, then one
    // byte for B for each pixel.
    SRGB = 1;

    // sRGBA, interleaved: one byte for R, one byte for G, one byte for B,
    // one byte for alpha or unused.
    SRGBA = 2;

    // Grayscale, one byte per pixel.
    GRAY8 = 3;

    // Grayscale, one uint16 per pixel.
    GRAY16 = 4;

    // YCbCr420P (1 bpp for Y, 0.25 bpp for U and V).
    // NOTE: NOT a valid ImageFrame format, but intended for
    // ScaleImageCalculatorOptions, VideoHeader, etc. to indicate that
    // YUVImage is used in place of ImageFrame.
    YCBCR420P = 5;

    // Similar to YCbCr420P, but the data is represented as the lower 10bits of
    // a uint16. Like YCbCr420P, this is NOT a valid ImageFrame, and the data is
    // carried within a YUVImage.
    YCBCR420P10 = 6;

    // sRGB, interleaved, each component is a uint16.
    SRGB48 = 7;

    // sRGBA, interleaved, each component is a uint16.
    SRGBA64 = 8;

    // One float per pixel.
    VEC32F1 = 9;

    // Two floats per pixel.
    VEC32F2 = 12;

    // LAB, interleaved: one byte for L, then one byte for a, then one
    // byte for b for each pixel.
    LAB8 = 10;

    // sBGRA, interleaved: one byte for B, one byte for G, one byte for R,
    // one byte for alpha or unused. This is the N32 format for Skia.
    SBGRA = 11;
  }
}
```