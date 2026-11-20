package nativewebp

import (
    //------------------------------
    //general
    //------------------------------
    "bytes"
    "encoding/binary"
    //------------------------------
    //imaging
    //------------------------------
    "image"
    "image/color"
    "image/draw"
    //------------------------------
    //testing
    //------------------------------
    "testing"
)

// testFrameSpec describes a single frame for hand-built container tests.
type testFrameSpec struct {
    img        image.Image
    x, y       int
    duration   uint32
    disposal   uint8
    blending   uint8
    extra      []byte // extra sub-chunks (e.g. ALPH) inserted before VP8L
    badPayload []byte // when non-nil, written instead of a VP8L sub-chunk
}

// encodeVP8LBytes encodes img as a bare VP8L RIFF file and returns the
// VP8L sub-chunk bytes (fourCC + size + payload), ready to embed in ANMF.
func encodeVP8LBytes(t *testing.T, img image.Image) []byte {
    t.Helper()

    buf := &bytes.Buffer{}
    if err := Encode(buf, img, nil); err != nil {
        t.Fatalf("Encode failed: %v", err)
    }

    data := buf.Bytes()
    // RIFF(12) + "VP8L" + size(4) + payload
    if len(data) < 20 || string(data[12:16]) != "VP8L" {
        t.Fatalf("expected VP8L chunk in encoded output")
    }
    return data[12:]
}

// buildAnimatedWebP assembles an animated WebP container from raw frame
// specs, giving tests full control over offsets, flags and chunk layout.
func buildAnimatedWebP(t *testing.T, width, height int, bg uint32, loop uint16, frames []testFrameSpec) []byte {
    t.Helper()

    body := &bytes.Buffer{}

    // VP8X: animation + alpha flags.
    body.Write([]byte("VP8X"))
    binary.Write(body, binary.LittleEndian, uint32(10))
    body.Write([]byte{0x12, 0x00, 0x00, 0x00})
    put24(body, uint32(width-1))
    put24(body, uint32(height-1))

    // ANIM.
    body.Write([]byte("ANIM"))
    binary.Write(body, binary.LittleEndian, uint32(6))
    binary.Write(body, binary.LittleEndian, bg)
    binary.Write(body, binary.LittleEndian, loop)

    for _, f := range frames {
        frameBody := &bytes.Buffer{}

        put24(frameBody, uint32(f.x/2))
        put24(frameBody, uint32(f.y/2))
        put24(frameBody, uint32(f.img.Bounds().Dx()-1))
        put24(frameBody, uint32(f.img.Bounds().Dy()-1))
        put24(frameBody, f.duration)
        // Wire layout used by this package: bit 0 disposal, bit 1 blending.
        frameBody.WriteByte((f.disposal & 1) | ((f.blending & 1) << 1))

        if f.badPayload != nil {
            frameBody.Write(f.badPayload)
        } else {
            if f.extra != nil {
                frameBody.Write(f.extra)
            }
            frameBody.Write(encodeVP8LBytes(t, f.img))
        }

        body.Write([]byte("ANMF"))
        binary.Write(body, binary.LittleEndian, uint32(frameBody.Len()))
        body.Write(frameBody.Bytes())
    }

    out := &bytes.Buffer{}
    out.Write([]byte("RIFF"))
    binary.Write(out, binary.LittleEndian, uint32(4+body.Len()))
    out.Write([]byte("WEBP"))
    out.Write(body.Bytes())

    return out.Bytes()
}

func put24(buf *bytes.Buffer, v uint32) {
    buf.Write([]byte{byte(v), byte(v >> 8), byte(v >> 16)})
}

func solidNRGBA(w, h int, c color.NRGBA) *image.NRGBA {
    img := image.NewNRGBA(image.Rect(0, 0, w, h))
    for i := 0; i < len(img.Pix); i += 4 {
        img.Pix[i+0] = c.R
        img.Pix[i+1] = c.G
        img.Pix[i+2] = c.B
        img.Pix[i+3] = c.A
    }
    return img
}

// compositeReference is an independent frame compositor implementing the
// WebP blending/disposal rules. It uses the standard library image/draw
// operators as ground truth, mirroring the decoder's own compositing.
func compositeReference(canvasW, canvasH int, bg color.NRGBA, frames []Frame) []*image.NRGBA {
    canvas := image.NewNRGBA(image.Rect(0, 0, canvasW, canvasH))
    fillNRGBA(canvas, bg)

    results := make([]*image.NRGBA, 0, len(frames))

    for _, f := range frames {
        r := f.FrameRect
        op := draw.Over
        if f.Blending == 1 {
            op = draw.Src
        }
        draw.Draw(canvas, r, f.FrameImage, r.Min, op)

        snap := image.NewNRGBA(canvas.Bounds())
        copy(snap.Pix, canvas.Pix)
        results = append(results, snap)

        if f.Disposal == 1 {
            clearNRGBA(canvas, r, bg)
        }
    }

    return results
}

func nrgbaEqualPix(a, b *image.NRGBA) bool {
    return bytes.Equal(a.Pix, b.Pix) && a.Rect.Eq(b.Rect) && a.Stride == b.Stride
}

func TestDecodeAnimationRoundTrip(t *testing.T) {
    canvasW, canvasH := 8, 8

    // Partial frames at even offsets with semi-transparent pixels, mixed
    // keep/clear disposal and per-frame timing. All frames use even offsets
    // because ANMF stores X/2 and Y/2.
    frameA := solidNRGBA(4, 4, color.NRGBA{200, 40, 40, 128})
    frameB := solidNRGBA(2, 6, color.NRGBA{40, 200, 40, 200})
    frameC := solidNRGBA(6, 2, color.NRGBA{40, 40, 200, 64})

    ani := &Animation{
        Images: []image.Image{
            repositionNRGBA(frameA, image.Rect(2, 2, 6, 6)),
            repositionNRGBA(frameB, image.Rect(0, 0, 2, 6)),
            repositionNRGBA(frameC, image.Rect(2, 6, 8, 8)),
        },
        Durations:       []uint{100, 200, 70},
        Disposals:       []uint{0, 1, 0},
        LoopCount:       3,
        BackgroundColor: 0xff112233,
    }

    buf := &bytes.Buffer{}
    if err := EncodeAll(buf, ani, nil); err != nil {
        t.Fatalf("EncodeAll failed: %v", err)
    }

    result, err := DecodeAnimation(bytes.NewReader(buf.Bytes()))
    if err != nil {
        t.Fatalf("DecodeAnimation failed: %v", err)
    }

    if result.LoopCount != 3 {
        t.Errorf("expected loop count 3 got %d", result.LoopCount)
    }
    if result.BackgroundColor != 0xff112233 {
        t.Errorf("expected background 0xff112233 got 0x%08x", result.BackgroundColor)
    }
    if result.Bounds != image.Rect(0, 0, canvasW, canvasH) {
        t.Errorf("expected canvas 8x8 got %v", result.Bounds)
    }
    if len(result.Frames) != 3 {
        t.Fatalf("expected 3 frames got %d", len(result.Frames))
    }

    expectedDurations := []uint32{100, 200, 70}
    expectedRects := []image.Rectangle{
        image.Rect(2, 2, 6, 6),
        image.Rect(0, 0, 2, 6),
        image.Rect(2, 6, 8, 8),
    }
    expectedDisposal := []uint8{0, 1, 0}

    for i, f := range result.Frames {
        if f.Duration != expectedDurations[i] {
            t.Errorf("frame %d: expected duration %d got %d", i, expectedDurations[i], f.Duration)
        }
        if f.FrameRect != expectedRects[i] {
            t.Errorf("frame %d: expected rect %v got %v", i, expectedRects[i], f.FrameRect)
        }
        if f.Disposal != expectedDisposal[i] {
            t.Errorf("frame %d: expected disposal %d got %d", i, expectedDisposal[i], f.Disposal)
        }
        if f.Image.Bounds() != result.Bounds {
            t.Errorf("frame %d: composited image is not full canvas", i)
        }
    }

    // Independent compositing must match every composited frame exactly.
    bg := color.NRGBA{R: 0x11, G: 0x22, B: 0x33, A: 0xff}
    reference := compositeReference(canvasW, canvasH, bg, result.Frames)
    for i, want := range reference {
        got := result.Frames[i].Image.(*image.NRGBA)
        if !nrgbaEqualPix(want, got) {
            t.Errorf("frame %d: composited pixels do not match reference", i)
        }
    }
}

func TestDecodeAnimationBlendingAndDisposal(t *testing.T) {
    canvasW, canvasH := 6, 6
    bg := color.NRGBA{0, 0, 0, 0}

    frameA := solidNRGBA(2, 2, color.NRGBA{255, 0, 0, 128}) // semi-transparent red
    frameB := solidNRGBA(2, 2, color.NRGBA{0, 255, 0, 255}) // opaque green, source-copy
    frameC := solidNRGBA(2, 2, color.NRGBA{0, 0, 255, 255}) // opaque blue

    data := buildAnimatedWebP(t, canvasW, canvasH, 0x00000000, 0, []testFrameSpec{
        {img: frameA, x: 2, y: 2, duration: 50, disposal: 0, blending: 0}, // alpha blend, keep
        {img: frameB, x: 2, y: 2, duration: 60, disposal: 1, blending: 1}, // no blend, clear
        {img: frameC, x: 0, y: 0, duration: 70, disposal: 0, blending: 0}, // alpha blend, keep
    })

    result, err := DecodeAnimation(bytes.NewReader(data))
    if err != nil {
        t.Fatalf("DecodeAnimation failed: %v", err)
    }
    if len(result.Frames) != 3 {
        t.Fatalf("expected 3 frames got %d", len(result.Frames))
    }

    // Frame 0: semi-transparent red over transparent background at (2,2).
    f0 := result.Frames[0].Image.(*image.NRGBA)
    if got := f0.NRGBAAt(2, 2); got != (color.NRGBA{255, 0, 0, 128}) {
        t.Errorf("frame0 (2,2): expected semi-transparent red got %v", got)
    }
    if got := f0.NRGBAAt(0, 0); got != bg {
        t.Errorf("frame0 (0,0): expected transparent background got %v", got)
    }

    // Frame 1: opaque green source-copied at (2,2); prior alpha ignored.
    f1 := result.Frames[1].Image.(*image.NRGBA)
    if got := f1.NRGBAAt(2, 2); got != (color.NRGBA{0, 255, 0, 255}) {
        t.Errorf("frame1 (2,2): expected opaque green got %v", got)
    }
    if got := f1.NRGBAAt(3, 3); got != (color.NRGBA{0, 255, 0, 255}) {
        t.Errorf("frame1 (3,3): expected opaque green got %v", got)
    }

    // After frame 1's clear disposal, (2,2) region must be transparent again.
    // Frame 2 only paints (0,0)-(2,2), so (2,2) stays cleared.
    f2 := result.Frames[2].Image.(*image.NRGBA)
    if got := f2.NRGBAAt(2, 2); got != bg {
        t.Errorf("frame2 (2,2): expected cleared transparent got %v", got)
    }
    if got := f2.NRGBAAt(0, 0); got != (color.NRGBA{0, 0, 255, 255}) {
        t.Errorf("frame2 (0,0): expected opaque blue got %v", got)
    }
}

func TestDecodeAnimationSemiTransparentOverOpaque(t *testing.T) {
    base := solidNRGBA(4, 4, color.NRGBA{0, 0, 0, 255})
    overlay := solidNRGBA(4, 4, color.NRGBA{255, 255, 255, 128})

    data := buildAnimatedWebP(t, 4, 4, 0x00000000, 0, []testFrameSpec{
        {img: base, x: 0, y: 0, duration: 40, disposal: 0, blending: 1},
        {img: overlay, x: 0, y: 0, duration: 40, disposal: 0, blending: 0},
    })

    result, err := DecodeAnimation(bytes.NewReader(data))
    if err != nil {
        t.Fatalf("DecodeAnimation failed: %v", err)
    }

    f1 := result.Frames[1].Image.(*image.NRGBA)
    // white at 128/255 over black -> 128,128,128 alpha 255
    want := color.NRGBA{128, 128, 128, 255}
    if got := f1.NRGBAAt(1, 1); got != want {
        t.Errorf("expected blended %v got %v", want, got)
    }
}

func TestDecodeAnimationLoopCountPreserved(t *testing.T) {
    frame := solidNRGBA(2, 2, color.NRGBA{10, 20, 30, 255})

    for _, loop := range []uint16{0, 1, 7, 65535} {
        data := buildAnimatedWebP(t, 2, 2, 0x00000000, loop, []testFrameSpec{
            {img: frame, x: 0, y: 0, duration: 33, disposal: 0, blending: 1},
        })

        result, err := DecodeAnimation(bytes.NewReader(data))
        if err != nil {
            t.Fatalf("loop %d: decode failed: %v", loop, err)
        }
        if result.LoopCount != loop {
            t.Errorf("expected loop %d got %d", loop, result.LoopCount)
        }
    }
}

func TestDecodeAnimationReEncodeRoundTrip(t *testing.T) {
    canvasW, canvasH := 6, 6
    frameA := solidNRGBA(2, 2, color.NRGBA{255, 0, 0, 128})
    frameB := solidNRGBA(2, 2, color.NRGBA{0, 255, 0, 255})

    data := buildAnimatedWebP(t, canvasW, canvasH, 0x00aabbcc, 5, []testFrameSpec{
        {img: frameA, x: 2, y: 2, duration: 90, disposal: 0, blending: 0},
        {img: frameB, x: 0, y: 0, duration: 110, disposal: 1, blending: 1},
    })

    first, err := DecodeAnimation(bytes.NewReader(data))
    if err != nil {
        t.Fatalf("first decode failed: %v", err)
    }

    reEncoded := &bytes.Buffer{}
    if err := EncodeAll(reEncoded, first.ToAnimation(), nil); err != nil {
        t.Fatalf("re-encode failed: %v", err)
    }

    second, err := DecodeAnimation(bytes.NewReader(reEncoded.Bytes()))
    if err != nil {
        t.Fatalf("second decode failed: %v", err)
    }

    if second.LoopCount != first.LoopCount || second.BackgroundColor != first.BackgroundColor {
        t.Errorf("metadata lost on round trip: loop %d->%d bg 0x%08x->0x%08x",
            first.LoopCount, second.LoopCount, first.BackgroundColor, second.BackgroundColor)
    }
    if len(second.Frames) != len(first.Frames) {
        t.Fatalf("frame count changed: %d -> %d", len(first.Frames), len(second.Frames))
    }

    for i := range first.Frames {
        if second.Frames[i].Duration != first.Frames[i].Duration ||
            second.Frames[i].Disposal != first.Frames[i].Disposal ||
            second.Frames[i].FrameRect != first.Frames[i].FrameRect {
            t.Errorf("frame %d metadata changed on round trip", i)
        }
        want := first.Frames[i].Image.(*image.NRGBA)
        got := second.Frames[i].Image.(*image.NRGBA)
        if !nrgbaEqualPix(want, got) {
            t.Errorf("frame %d pixels changed on round trip", i)
        }
    }
}

func TestDecodeAnimationStaticDecodeUnchanged(t *testing.T) {
    // The static Decode path must remain untouched and still reject
    // animated containers as before.
    img := generateTestImageNRGBA(8, 8, 64, true)
    buf := &bytes.Buffer{}
    if err := Encode(buf, img, nil); err != nil {
        t.Fatalf("Encode failed: %v", err)
    }

    static, err := Decode(bytes.NewReader(buf.Bytes()))
    if err != nil {
        t.Fatalf("static Decode failed: %v", err)
    }
    if static.Bounds() != img.Bounds() {
        t.Errorf("static Decode bounds changed: %v", static.Bounds())
    }

    // And DecodeAnimation must refuse a static file.
    if _, err := DecodeAnimation(bytes.NewReader(buf.Bytes())); err == nil {
        t.Error("expected error decoding static WebP as animation, got nil")
    }
}

func TestDecodeAnimationErrors(t *testing.T) {
    frame := solidNRGBA(2, 2, color.NRGBA{1, 2, 3, 255})

    build := func(frames []testFrameSpec) []byte {
        return buildAnimatedWebP(t, 6, 6, 0, 0, frames)
    }

    for _, tt := range []struct {
        name        string
        data        []byte
        expectedErr string
    }{
        {
            name:        "not webp",
            data:        []byte("not a WebP file at all!!!"),
            expectedErr: "riff: missing RIFF chunk header",
        },
        {
            name:        "riff only",
            data:        []byte("RIFF\x04\x00\x00\x00WEBP"),
            expectedErr: "webp: missing VP8X chunk",
        },
        {
            name: "bad riff size",
            data: func() []byte {
                d := build(nil)
                binary.LittleEndian.PutUint32(d[4:8], 1)
                return d
            }(),
            expectedErr: "webp: invalid RIFF size",
        },
        {
            name: "truncated before chunk header",
            data: func() []byte {
                d := build(nil) // VP8X + ANIM, no frames
                return d[:14]
            }(),
            expectedErr: "webp: missing VP8X chunk",
        },
        {
            name: "truncated inside frame payload",
            data: func() []byte {
                // Build an ANMF whose outer size is consistent with the file
                // but whose inner VP8L sub-chunk claims more bytes than the
                // frame payload actually holds.
                body := &bytes.Buffer{}
                body.Write([]byte("VP8X"))
                binary.Write(body, binary.LittleEndian, uint32(10))
                body.Write([]byte{0x02, 0, 0, 0})
                put24(body, 1)
                put24(body, 1)
                body.Write([]byte("ANIM"))
                binary.Write(body, binary.LittleEndian, uint32(6))
                body.Write([]byte{0, 0, 0, 0, 0, 0})

                frameBody := &bytes.Buffer{}
                put24(frameBody, 0)
                put24(frameBody, 0)
                put24(frameBody, 0)
                put24(frameBody, 0)
                put24(frameBody, 40)
                frameBody.WriteByte(1)
                // Inner VP8L header declaring 100 bytes but only 4 follow.
                frameBody.Write([]byte("VP8L"))
                binary.Write(frameBody, binary.LittleEndian, uint32(100))
                frameBody.Write([]byte{0x2f, 0, 0, 0})

                body.Write([]byte("ANMF"))
                binary.Write(body, binary.LittleEndian, uint32(frameBody.Len()))
                body.Write(frameBody.Bytes())

                out := &bytes.Buffer{}
                out.Write([]byte("RIFF"))
                binary.Write(out, binary.LittleEndian, uint32(4+body.Len()))
                out.Write([]byte("WEBP"))
                out.Write(body.Bytes())
                return out.Bytes()
            }(),
            expectedErr: "webp: truncated frame chunk",
        },
        {
            name: "bad top-level chunk length",
            data: func() []byte {
                d := build(nil)
                // ANIM lives at offset 30; inflate its size beyond the file.
                binary.LittleEndian.PutUint32(d[34:38], 0x7fffffff)
                return d
            }(),
            expectedErr: "webp: invalid chunk length",
        },
        {
            name: "anmf without anim/vp8x",
            data: func() []byte {
                body := &bytes.Buffer{}
                body.Write([]byte("ANMF"))
                binary.Write(body, binary.LittleEndian, uint32(20))
                body.Write(make([]byte, 20))
                out := &bytes.Buffer{}
                out.Write([]byte("RIFF"))
                binary.Write(out, binary.LittleEndian, uint32(4+body.Len()))
                out.Write([]byte("WEBP"))
                out.Write(body.Bytes())
                return out.Bytes()
            }(),
            expectedErr: "webp: ANMF chunk without canvas",
        },
        {
            name: "frame outside canvas",
            data: build([]testFrameSpec{
                {img: solidNRGBA(4, 4, color.NRGBA{0, 0, 0, 255}), x: 4, y: 4, duration: 40, blending: 1},
            }),
            expectedErr: "webp: frame rectangle outside canvas",
        },
        {
            name: "zero duration",
            data: build([]testFrameSpec{
                {img: frame, x: 0, y: 0, duration: 0, blending: 1},
            }),
            expectedErr: "webp: frame duration must be greater than zero",
        },
        {
            name:        "anim with no frames",
            data:        build(nil),
            expectedErr: "webp: animation contains no frames",
        },
        {
            name: "frame without image payload",
            data: build([]testFrameSpec{
                {img: frame, x: 0, y: 0, duration: 40, blending: 1, badPayload: []byte("JUNK\x04\x00\x00\x00abcd")},
            }),
            expectedErr: "webp: frame contains no image data",
        },
        {
            name: "frame with corrupt vp8l",
            data: build([]testFrameSpec{
                {img: frame, x: 0, y: 0, duration: 40, blending: 1, badPayload: []byte("VP8L\x04\x00\x00\x00\x00\x00\x00\x00")},
            }),
            expectedErr: "vp8l: invalid header",
        },
        {
            name: "alph chunk rejected",
            data: build([]testFrameSpec{
                {
                    img:      frame,
                    x:        0,
                    y:        0,
                    duration: 40,
                    blending: 1,
                    // ALPH with 1-byte payload plus mandatory even-size padding.
                    extra: []byte("ALPH\x01\x00\x00\x00\xff\x00"),
                },
            }),
            expectedErr: "webp: ALPH chunks are not supported",
        },
        {
            name: "canvas too large",
            data: func() []byte {
                body := &bytes.Buffer{}
                body.Write([]byte("VP8X"))
                binary.Write(body, binary.LittleEndian, uint32(10))
                body.Write([]byte{0x02, 0, 0, 0})
                put24(body, 1<<14) // width = 16385
                put24(body, 0)
                out := &bytes.Buffer{}
                out.Write([]byte("RIFF"))
                binary.Write(out, binary.LittleEndian, uint32(4+body.Len()))
                out.Write([]byte("WEBP"))
                out.Write(body.Bytes())
                return out.Bytes()
            }(),
            expectedErr: "webp: animation canvas too large",
        },
        {
            name: "static extended vp8x without animation flag",
            data: func() []byte {
                buf := &bytes.Buffer{}
                if err := Encode(buf, frame, &Options{UseExtendedFormat: true}); err != nil {
                    t.Fatalf("Encode failed: %v", err)
                }
                return buf.Bytes()
            }(),
            expectedErr: "webp: not an animated WebP",
        },
        {
            name: "duplicate vp8x",
            data: func() []byte {
                d := build([]testFrameSpec{
                    {img: frame, x: 0, y: 0, duration: 40, blending: 1},
                })
                // Insert a second VP8X right after the first one (offset 12).
                vp8x := d[12:30]
                return append(d[:30], append(vp8x, d[30:]...)...)
            }(),
            expectedErr: "webp: duplicate VP8X chunk",
        },
    } {
        _, err := DecodeAnimation(bytes.NewReader(tt.data))

        if tt.expectedErr == "" {
            if err != nil {
                t.Errorf("%s: expected success got error %v", tt.name, err)
            }
            continue
        }

        if err == nil {
            t.Errorf("%s: expected error %q got nil", tt.name, tt.expectedErr)
            continue
        }
        if err.Error() != tt.expectedErr {
            t.Errorf("%s: expected error %q got %q", tt.name, tt.expectedErr, err.Error())
        }
    }
}

func TestDecodeAnimationInputTooLarge(t *testing.T) {
    big := bytes.Repeat([]byte{0}, maxAnimationInput+1)
    big[0], big[1], big[2], big[3] = 'R', 'I', 'F', 'F'

    _, err := DecodeAnimation(bytes.NewReader(big))
    if err == nil || err.Error() != "webp: input too large" {
        t.Errorf("expected input-too-large error got %v", err)
    }
}

func TestDecodeAnimationFrameCountLimit(t *testing.T) {
    frame := solidNRGBA(1, 1, color.NRGBA{0, 0, 0, 255})
    vp8lChunk := encodeVP8LBytes(t, frame) // reuse one payload for all frames

    body := &bytes.Buffer{}
    body.Write([]byte("VP8X"))
    binary.Write(body, binary.LittleEndian, uint32(10))
    body.Write([]byte{0x02, 0, 0, 0})
    put24(body, 0)
    put24(body, 0)
    body.Write([]byte("ANIM"))
    binary.Write(body, binary.LittleEndian, uint32(6))
    body.Write([]byte{0, 0, 0, 0, 0, 0})

    frameBody := &bytes.Buffer{}
    put24(frameBody, 0) // x/2
    put24(frameBody, 0) // y/2
    put24(frameBody, 0) // width-1
    put24(frameBody, 0) // height-1
    put24(frameBody, 10)
    frameBody.WriteByte(1) // no blending
    frameBody.Write(vp8lChunk)

    for i := 0; i < maxAnimationFrames+1; i++ {
        body.Write([]byte("ANMF"))
        binary.Write(body, binary.LittleEndian, uint32(frameBody.Len()))
        body.Write(frameBody.Bytes())
    }

    out := &bytes.Buffer{}
    out.Write([]byte("RIFF"))
    binary.Write(out, binary.LittleEndian, uint32(4+body.Len()))
    out.Write([]byte("WEBP"))
    out.Write(body.Bytes())

    _, err := DecodeAnimation(bytes.NewReader(out.Bytes()))
    if err == nil || err.Error() != "webp: too many animation frames" {
        t.Errorf("expected frame-limit error got %v", err)
    }
}

func TestDecodeAnimationUnknownChunksIgnored(t *testing.T) {
    frame := solidNRGBA(2, 2, color.NRGBA{9, 8, 7, 255})
    data := buildAnimatedWebP(t, 2, 2, 0, 1, []testFrameSpec{
        {img: frame, x: 0, y: 0, duration: 40, blending: 1},
    })

    // Insert an odd-length EXIF chunk (5-byte payload + 1 padding byte)
    // between VP8X and ANIM.
    exif := make([]byte, 0, 14)
    exif = append(exif, []byte("EXIF")...)
    exif = append(exif, 5, 0, 0, 0)
    exif = append(exif, 'h', 'e', 'l', 'l', 'o') // 5 payload bytes
    exif = append(exif, 0)                       // even-size padding
    data = append(data[:30], append(exif, data[30:]...)...)
    binary.LittleEndian.PutUint32(data[4:8], uint32(len(data)-8))

    result, err := DecodeAnimation(bytes.NewReader(data))
    if err != nil {
        t.Fatalf("expected unknown chunk to be ignored, got %v", err)
    }
    if len(result.Frames) != 1 {
        t.Errorf("expected 1 frame got %d", len(result.Frames))
    }
}

func TestAnimatedImageToAnimation(t *testing.T) {
    frameA := solidNRGBA(2, 2, color.NRGBA{255, 0, 0, 255})
    frameB := solidNRGBA(2, 2, color.NRGBA{0, 0, 255, 128})

    data := buildAnimatedWebP(t, 4, 4, 0x00102030, 2, []testFrameSpec{
        {img: frameA, x: 0, y: 0, duration: 80, disposal: 0, blending: 1},
        {img: frameB, x: 2, y: 2, duration: 90, disposal: 1, blending: 0},
    })

    result, err := DecodeAnimation(bytes.NewReader(data))
    if err != nil {
        t.Fatalf("decode failed: %v", err)
    }

    ani := result.ToAnimation()
    if len(ani.Images) != 2 || len(ani.Durations) != 2 || len(ani.Disposals) != 2 {
        t.Fatalf("converted slices have wrong length")
    }
    if ani.LoopCount != 2 || ani.BackgroundColor != 0x00102030 {
        t.Errorf("converted metadata mismatch: loop %d bg 0x%08x", ani.LoopCount, ani.BackgroundColor)
    }
    if ani.Durations[0] != 80 || ani.Durations[1] != 90 {
        t.Errorf("converted durations mismatch: %v", ani.Durations)
    }
    if ani.Disposals[0] != 0 || ani.Disposals[1] != 1 {
        t.Errorf("converted disposals mismatch: %v", ani.Disposals)
    }
    if ani.Images[0].Bounds().Min != (image.Point{0, 0}) ||
        ani.Images[1].Bounds().Min != (image.Point{2, 2}) {
        t.Errorf("converted frame offsets lost: %v %v",
            ani.Images[0].Bounds().Min, ani.Images[1].Bounds().Min)
    }
}
