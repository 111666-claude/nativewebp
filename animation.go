package nativewebp

import (
    //------------------------------
    //general
    //------------------------------
    "bytes"
    "encoding/binary"
    "io"
    //------------------------------
    //imaging
    //------------------------------
    "image"
    "image/color"
    "image/draw"
    //------------------------------
    //errors
    //------------------------------
    "errors"
    //------------------------------
    //decoders
    //------------------------------
    decoderWebP "golang.org/x/image/webp"
)

// Memory and structural limits applied while decoding an animation.
// These guard against malicious or corrupt files exhausting RAM.
const (
    // maxAnimationInput mirrors the static decoder read cap.
    maxAnimationInput = 256 * 1024 * 1024

    // maxAnimationDimension is the largest canvas side accepted
    // (the WebP container stores dimensions as 24-bit values).
    maxAnimationDimension = 1 << 14

    // maxAnimationFrames bounds the number of ANMF chunks accepted.
    maxAnimationFrames = 1 << 16

    // maxAnimationPixels bounds the total number of NRGBA pixels
    // retained for the decoded result (canvas + every frame snapshot).
    // 1<<26 pixels * 4 bytes = 256 MiB of decoded pixel memory.
    maxAnimationPixels = 1 << 26
)

// Frame is a single decoded animation frame.
//
// Fields:
//   - Image: The fully composited canvas at the moment this frame is
//     displayed, after blending and any prior disposal. Every image has
//     the full canvas rectangle, so frames can be inspected directly.
//   - FrameImage: The frame's own (sub-rectangle) image positioned at its
//     canvas offset. It is the decoded VP8/VP8L payload before compositing.
//   - FrameRect: The frame rectangle within the canvas.
//   - Duration: Frame display time in milliseconds.
//   - Disposal: Disposal method applied after display; 0 = keep,
//     1 = clear the frame rectangle to the background color.
//   - Blending: Blending method used while compositing; 0 = alpha
//     (source-over) blending, 1 = no blending (source copy).
type Frame struct {
    Image      image.Image
    FrameImage image.Image
    FrameRect  image.Rectangle
    Duration   uint32
    Disposal   uint8
    Blending   uint8
}

// AnimatedImage holds the decoded result of an animated WebP file.
//
// Fields:
//   - Frames: Decoded frames in display order.
//   - Bounds: Canvas rectangle (always starting at 0,0).
//   - LoopCount: Number of times the animation repeats; 0 means infinite.
//   - BackgroundColor: Canvas background color as stored in the ANIM chunk,
//     packed as 0x00RRGGBB. Used for transparent areas and clear disposal.
type AnimatedImage struct {
    Frames          []Frame
    Bounds          image.Rectangle
    LoopCount       uint16
    BackgroundColor uint32
}

// ToAnimation converts the decoded animation into the encoder-facing
// Animation structure. The per-frame sub-rectangle images retain their
// canvas offsets, so re-encoding with EncodeAll preserves frame placement,
// durations, disposal, loop count and background color.
func (a *AnimatedImage) ToAnimation() *Animation {
    images := make([]image.Image, len(a.Frames))
    durations := make([]uint, len(a.Frames))
    disposals := make([]uint, len(a.Frames))

    for i, frame := range a.Frames {
        images[i] = frame.FrameImage
        durations[i] = uint(frame.Duration)
        disposals[i] = uint(frame.Disposal)
    }

    return &Animation{
        Images:          images,
        Durations:       durations,
        Disposals:       disposals,
        LoopCount:       a.LoopCount,
        BackgroundColor: a.BackgroundColor,
        CanvasWidth:     a.Bounds.Dx(),
        CanvasHeight:    a.Bounds.Dy(),
    }
}

// DecodeAnimation reads an animated WebP (VP8X container with ANIM/ANMF
// chunks) from r and returns every frame composited onto the full canvas.
//
// Frame payloads are decoded with the existing VP8/VP8L decoder; VP8L
// transparency is fully supported. Frames are composited according to the
// WebP specification, honoring per-frame offsets, durations, blending
// (alpha vs. source copy) and disposal (keep vs. clear to background).
//
// The static Decode behavior is unchanged; this function only accepts
// animated containers and returns an error for static or non-WebP input.
//
// Reads are capped at 256 MiB and decoded pixel memory is bounded, so
// corrupt or hostile files (bad chunk lengths, out-of-bounds frames, zero
// durations, truncated data, oversized canvases) produce an error rather
// than unbounded allocation.
func DecodeAnimation(r io.Reader) (*AnimatedImage, error) {
    data, err := io.ReadAll(io.LimitReader(r, maxAnimationInput+1))
    if err != nil {
        return nil, err
    }

    if len(data) > maxAnimationInput {
        return nil, errors.New("webp: input too large")
    }

    return decodeAnimation(data)
}

func decodeAnimation(data []byte) (*AnimatedImage, error) {
    // RIFF header: "RIFF", 32-bit size, "WEBP".
    if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
        return nil, errors.New("riff: missing RIFF chunk header")
    }

    riffSize := int(binary.LittleEndian.Uint32(data[4:8])) + 8
    if riffSize < 12 {
        return nil, errors.New("webp: invalid RIFF size")
    }

    // Never walk past the declared RIFF payload or the available data.
    riffEnd := riffSize
    if riffEnd > len(data) {
        riffEnd = len(data)
    }

    var (
        canvas           *image.NRGBA
        canvasW, canvasH int
        bgColor          color.NRGBA
        bgPacked         uint32
        loopCount        uint16
        frames           []Frame
        gotVP8X, gotANIM bool
        pixelsUsed       uint64
        frameCount       int
    )

    pos := 12
    for pos+8 <= riffEnd {
        fourCC := string(data[pos : pos+4])
        chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
        body := pos + 8

        // Guard against chunk sizes that overflow the file or int range.
        if chunkSize < 0 || body > riffEnd || chunkSize > riffEnd-body {
            return nil, errors.New("webp: invalid chunk length")
        }

        chunkEnd := body + chunkSize
        if chunkEnd > riffEnd {
            return nil, errors.New("webp: truncated chunk")
        }

        switch fourCC {
        case "VP8X":
            if gotVP8X {
                return nil, errors.New("webp: duplicate VP8X chunk")
            }
            if chunkSize != 10 {
                return nil, errors.New("webp: invalid VP8X chunk length")
            }

            flags := data[body]
            if flags&0x02 == 0 {
                return nil, errors.New("webp: not an animated WebP")
            }

            canvasW = int(uint24(data[body+4:])) + 1
            canvasH = int(uint24(data[body+7:])) + 1

            if canvasW > maxAnimationDimension || canvasH > maxAnimationDimension {
                return nil, errors.New("webp: animation canvas too large")
            }

            gotVP8X = true

        case "ANIM":
            if !gotVP8X {
                return nil, errors.New("webp: ANIM chunk before VP8X")
            }
            if gotANIM {
                return nil, errors.New("webp: duplicate ANIM chunk")
            }
            if chunkSize != 6 {
                return nil, errors.New("webp: invalid ANIM chunk length")
            }

            bgPacked = binary.LittleEndian.Uint32(data[body : body+4])
            bgColor = color.NRGBA{
                R: uint8(bgPacked >> 16),
                G: uint8(bgPacked >> 8),
                B: uint8(bgPacked),
                A: uint8(bgPacked >> 24),
            }
            loopCount = binary.LittleEndian.Uint16(data[body+4 : body+6])
            gotANIM = true

            canvasPixels, ok := mulUint32(canvasW, canvasH)
            if !ok || canvasPixels > maxAnimationPixels {
                return nil, errors.New("webp: animation canvas too large")
            }
            pixelsUsed += canvasPixels

            canvas = image.NewNRGBA(image.Rect(0, 0, canvasW, canvasH))
            fillNRGBA(canvas, bgColor)

        case "ANMF":
            if !gotVP8X || !gotANIM || canvas == nil {
                return nil, errors.New("webp: ANMF chunk without canvas")
            }

            frameCount++
            if frameCount > maxAnimationFrames {
                return nil, errors.New("webp: too many animation frames")
            }

            if chunkSize < 16 {
                return nil, errors.New("webp: invalid ANMF chunk length")
            }

            // 3 bytes X, 3 bytes Y, 3 bytes width-1, 3 bytes height-1,
            // 3 bytes duration, 1 byte flags.
            frameX := int(uint24(data[body:])) * 2
            frameY := int(uint24(data[body+3:])) * 2
            frameW := int(uint24(data[body+6:])) + 1
            frameH := int(uint24(data[body+9:])) + 1
            duration := uint24(data[body+12:])
            flags := data[body+15]

            if duration == 0 {
                return nil, errors.New("webp: frame duration must be greater than zero")
            }

            frameRect := image.Rect(frameX, frameY, frameX+frameW, frameY+frameH)
            if !frameRect.In(canvas.Bounds()) {
                return nil, errors.New("webp: frame rectangle outside canvas")
            }

            framePixels, ok := mulUint32(frameW, frameH)
            if !ok || framePixels > maxAnimationPixels {
                return nil, errors.New("webp: animation frame too large")
            }

            // Walk the ANMF payload for its VP8L/VP8 sub-chunk.
            payloadEnd := chunkEnd
            sub := body + 16
            var (
                frameImg   image.Image
                decodeErr  error
                sawPayload bool
                sawAlpha   bool
            )

            for sub+8 <= payloadEnd {
                subFourCC := string(data[sub : sub+4])
                subSize := int(binary.LittleEndian.Uint32(data[sub+4 : sub+8]))
                subBody := sub + 8

                if subSize < 0 || subBody > payloadEnd {
                    return nil, errors.New("webp: invalid frame chunk length")
                }

                subEnd := subBody + subSize
                if subEnd > payloadEnd {
                    return nil, errors.New("webp: truncated frame chunk")
                }

                switch subFourCC {
                case "VP8L", "VP8 ":
                    if sawPayload {
                        return nil, errors.New("webp: multiple image chunks in frame")
                    }

                    // The underlying decoder expects a full RIFF/WEBP
                    // container, so wrap the bare frame payload back into one.
                    wrapped := make([]byte, 12+8+subSize)
                    copy(wrapped[0:4], []byte("RIFF"))
                    binary.LittleEndian.PutUint32(wrapped[4:8], uint32(4+8+subSize))
                    copy(wrapped[8:12], []byte("WEBP"))
                    copy(wrapped[12:16], []byte(subFourCC))
                    binary.LittleEndian.PutUint32(wrapped[16:20], uint32(subSize))
                    copy(wrapped[20:], data[subBody:subEnd])

                    frameImg, decodeErr = decoderWebP.Decode(bytes.NewReader(wrapped))
                    if decodeErr != nil {
                        return nil, decodeErr
                    }
                    sawPayload = true

                case "ALPH":
                    // Alpha chunk only accompanies lossy VP8; lossless VP8L
                    // carries alpha internally and needs no handling here.
                    sawAlpha = true
                }

                sub = subEnd + (subSize & 1)
            }

            if !sawPayload {
                return nil, errors.New("webp: frame contains no image data")
            }

            if sawAlpha {
                return nil, errors.New("webp: ALPH chunks are not supported")
            }

            decodedBounds := frameImg.Bounds()
            if decodedBounds.Dx() != frameW || decodedBounds.Dy() != frameH {
                return nil, errors.New("webp: frame dimensions do not match ANMF header")
            }

            // Reposition the decoded sub-frame at its canvas offset while
            // retaining the original pixel buffer.
            positioned := repositionNRGBA(frameImg, frameRect)

            // NOTE: bit layout follows this package's encoder: bit 0 is the
            // disposal method and bit 1 is the blending method (the canonical
            // WebP layout swaps the two). Matching the writer keeps files
            // produced by EncodeAll/DecodeAnimation round-trippable.
            disposal := flags & 0x01
            blending := (flags >> 1) & 0x01

            op := draw.Over
            if blending == 1 {
                op = draw.Src
            }
            draw.Draw(canvas, frameRect, positioned, frameRect.Min, op)

            snapshotPixels, ok := mulUint32(canvasW, canvasH)
            if !ok {
                return nil, errors.New("webp: animation too large")
            }
            if pixelsUsed+framePixels+snapshotPixels > maxAnimationPixels {
                return nil, errors.New("webp: animation exceeds memory limit")
            }
            pixelsUsed += framePixels + snapshotPixels

            snapshot := image.NewNRGBA(canvas.Bounds())
            copy(snapshot.Pix, canvas.Pix)

            frames = append(frames, Frame{
                Image:      snapshot,
                FrameImage: positioned,
                FrameRect:  frameRect,
                Duration:   duration,
                Disposal:   disposal,
                Blending:   blending,
            })

            // Apply disposal after the frame has been displayed/snapshotted.
            if disposal == 1 {
                clearNRGBA(canvas, frameRect, bgColor)
            }

        default:
            // Ignore unknown/metadata top-level chunks (EXIF, XMP, ICCP, ...).
        }

        pos = chunkEnd + (chunkSize & 1)
    }

    if !gotVP8X {
        return nil, errors.New("webp: missing VP8X chunk")
    }
    if !gotANIM {
        return nil, errors.New("webp: missing ANIM chunk")
    }
    if len(frames) == 0 {
        return nil, errors.New("webp: animation contains no frames")
    }

    return &AnimatedImage{
        Frames:          frames,
        Bounds:          canvas.Bounds(),
        LoopCount:       loopCount,
        BackgroundColor: bgPacked,
    }, nil
}

func uint24(b []byte) uint32 {
    return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
}

// mulUint32 returns a*b with an overflow guard.
func mulUint32(a, b int) (uint64, bool) {
    if a <= 0 || b <= 0 {
        return 0, false
    }
    v := uint64(a) * uint64(b)
    if v > (1 << 40) {
        return 0, false
    }
    return v, true
}

func fillNRGBA(img *image.NRGBA, c color.NRGBA) {
    for i := 0; i < len(img.Pix); i += 4 {
        img.Pix[i+0] = c.R
        img.Pix[i+1] = c.G
        img.Pix[i+2] = c.B
        img.Pix[i+3] = c.A
    }
}

func clearNRGBA(img *image.NRGBA, rect image.Rectangle, c color.NRGBA) {
    for y := rect.Min.Y; y < rect.Max.Y; y++ {
        for x := rect.Min.X; x < rect.Max.X; x++ {
            i := img.PixOffset(x, y)
            img.Pix[i+0] = c.R
            img.Pix[i+1] = c.G
            img.Pix[i+2] = c.B
            img.Pix[i+3] = c.A
        }
    }
}

// repositionNRGBA returns an NRGBA sharing src's pixels but with bounds
// anchored at rect.Min, so the frame carries its canvas offset. The encoder
// reads pixels relative to the image bounds (see flatten), which supports
// non-zero origins.
func repositionNRGBA(src image.Image, rect image.Rectangle) *image.NRGBA {
    nrgba, ok := src.(*image.NRGBA)
    if !ok {
        converted := image.NewNRGBA(rect)
        draw.Draw(converted, rect, src, src.Bounds().Min, draw.Src)
        return converted
    }

    return &image.NRGBA{
        Pix:    nrgba.Pix,
        Stride: nrgba.Stride,
        Rect:   rect,
    }
}
