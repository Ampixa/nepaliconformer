import {
  AbsoluteFill,
  Audio,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export type Scene = {
  kind: "title" | "everyday" | "gap" | "modes" | "outro";
  id: string;
  durFrames: number;
  wav?: string;
  transcript?: string;
  veo?: string | null; // Gemini/Veo footage for the top viewport, if present
};

export type TimelineData = { scenes: Scene[] };

// house palette — matches asr.ampixa.com
const BG = "#FFFFFF";
const INK = "#111111";
const CRIMSON = "#DC143C";
const MUTED = "#555555";
const FAINT = "#888888";
const LINE = "#D3D3D3";
const OK = "#1B8A3A";
const MONO = "'Courier New', 'Lucida Console', Monaco, monospace";
const DEV = "'Noto Sans Devanagari', -apple-system, system-ui, sans-serif";

const Equalizer: React.FC<{ bars?: number; active: boolean }> = ({
  bars = 24,
  active,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 34 }}>
      {Array.from({ length: bars }).map((_, i) => {
        const h = active
          ? 5 + Math.abs(Math.sin(frame / 2.6 + i * 1.31)) * 26
          : 4;
        return (
          <div
            key={i}
            style={{
              width: 7,
              height: h,
              background: CRIMSON,
            }}
          />
        );
      })}
    </div>
  );
};

// The fixed bottom-35% ASR terminal. Transcript is the REAL model output for the
// narration audio playing in this scene.
const ASRBox: React.FC<{
  transcript: string;
  progress: number; // 0..1 across the scene
}> = ({ transcript, progress }) => {
  const frame = useCurrentFrame();
  const words = transcript.split(" ");
  const shown = Math.max(0, Math.floor(progress * words.length));
  const cursorOn = Math.floor(frame / 7) % 2 === 0 && progress < 1;
  const pct = Math.round(Math.min(progress, 1) * 100);
  const micOn = Math.floor(frame / 14) % 2 === 0;
  return (
    <div
      style={{
        height: "35%",
        background: BG,
        borderTop: `3px solid ${LINE}`,
        fontFamily: MONO,
        padding: "22px 46px",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        position: "relative",
      }}
    >
      <div style={{ display: "flex", gap: 36, fontSize: 21, color: FAINT }}>
        <span>
          <span style={{ color: CRIMSON, opacity: micOn ? 1 : 0.3 }}>[●]</span>{" "}
          MIC: <span style={{ color: OK }}>ACTIVE</span>
        </span>
        <span>
          MODE: <span style={{ color: INK }}>STREAMING (520ms)</span>
        </span>
        <span>
          LOC: <span style={{ color: INK }}>NEPAL</span>
        </span>
        <span style={{ marginLeft: "auto", color: FAINT }}>nepaliconformer v1</span>
      </div>
      <div
        style={{
          fontFamily: DEV,
          fontSize: 38,
          lineHeight: 1.55,
          color: INK,
          minHeight: 120,
        }}
      >
        <span style={{ color: FAINT, fontFamily: MONO }}>{">> "}</span>
        {words.slice(0, shown).join(" ")}
        {cursorOn ? <span style={{ color: CRIMSON }}>▌</span> : null}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 26 }}>
        <Equalizer active={progress < 0.98} />
        <div style={{ fontSize: 21, color: FAINT }}>
          [{"|".repeat(Math.round(pct / 8)).padEnd(13, " ")}] {pct}%
        </div>
      </div>
    </div>
  );
};

// ------------------------------------------------------------- top viewports

const Viewport: React.FC<{ veo?: string | null; children?: React.ReactNode }> = ({
  veo,
  children,
}) => (
  <div style={{ height: "65%", position: "relative", overflow: "hidden", background: BG }}>
    {veo ? (
      <OffthreadVideo
        src={staticFile(veo)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
        muted
      />
    ) : null}
    <div style={{ position: "absolute", inset: 0 }}>{children}</div>
  </div>
);

const PixelHills: React.FC = () => (
  // procedural fallback backdrop: layered pixel ridgelines, light greys on white
  <AbsoluteFill>
    <div
      style={{
        position: "absolute",
        bottom: 0,
        width: "100%",
        height: "45%",
        background: "#F0F0F0",
        clipPath:
          "polygon(0 62%, 9% 44%, 17% 58%, 28% 30%, 38% 55%, 50% 22%, 61% 50%, 72% 34%, 84% 56%, 100% 40%, 100% 100%, 0 100%)",
      }}
    />
    <div
      style={{
        position: "absolute",
        bottom: 0,
        width: "100%",
        height: "30%",
        background: "#E2E2E2",
        clipPath:
          "polygon(0 70%, 12% 46%, 25% 66%, 40% 40%, 55% 64%, 70% 44%, 86% 62%, 100% 50%, 100% 100%, 0 100%)",
      }}
    />
  </AbsoluteFill>
);

const Card: React.FC<{
  title: string;
  color: string;
  lines: string[];
  x: number;
  enterFrame: number;
  width?: number;
}> = ({ title, color, lines, x, enterFrame, width = 560 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - enterFrame, fps, config: { damping: 15 } });
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: 90,
        width,
        transform: `translateY(${(1 - s) * 60}px)`,
        opacity: s,
        background: "#FFFFFFF2",
        border: `2px solid ${color}`,
        padding: "26px 30px",
        fontFamily: MONO,
      }}
    >
      <div style={{ fontSize: 27, color, fontWeight: 700, marginBottom: 14 }}>{title}</div>
      {lines.map((l) => (
        <div key={l} style={{ fontSize: 22, color: MUTED, lineHeight: 1.8 }}>
          {l}
        </div>
      ))}
    </div>
  );
};

// scene 0: title card
const Title: React.FC<{ dur: number }> = ({ dur }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pop = spring({ frame: frame - 4, fps, config: { damping: 13 } });
  const sub = interpolate(frame, [dur * 0.35, dur * 0.6], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill
      style={{
        background: BG,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 30,
      }}
    >
      <div
        style={{
          fontFamily: MONO,
          fontWeight: 700,
          fontSize: 124,
          letterSpacing: -3,
          color: INK,
          transform: `scale(${pop})`,
        }}
      >
        nepali<span style={{ color: CRIMSON }}>conformer</span>
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 40,
          color: CRIMSON,
          opacity: sub,
        }}
      >
        ASR MODEL FOR NEPALI
      </div>
      <div style={{ fontFamily: DEV, fontSize: 34, color: MUTED, opacity: sub }}>
        नेपालको लागि बनेको स्वचालित बोली पहिचान
      </div>
      <MicPrompt dur={dur} />
    </AbsoluteFill>
  );
};

const MicPrompt: React.FC<{ dur: number }> = ({ dur }) => {
  const frame = useCurrentFrame();
  const on = interpolate(frame, [dur * 0.45, dur * 0.6], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const blink = Math.floor(frame / 10) % 2 === 0;
  return (
    <div
      style={{
        marginTop: 26,
        opacity: on,
        display: "flex",
        alignItems: "center",
        gap: 18,
        fontFamily: MONO,
        fontSize: 34,
        color: INK,
      }}
    >
      <span style={{ fontSize: 40 }}>🎙</span>
      <span>
        <span style={{ color: CRIMSON, opacity: blink ? 1 : 0.25 }}>[●]</span> MIC: ACTIVE —{" "}
        <span style={{ fontFamily: DEV }}>बोल्न सुरु गर्नुहोस्</span>
        <span style={{ color: FAINT }}> · start speaking</span>
        {blink ? <span style={{ color: CRIMSON }}> ▌</span> : null}
      </span>
    </div>
  );
};

// scene 1: everyday Nepal
const Everyday: React.FC<{ s: Scene }> = ({ s }) => {
  const frame = useCurrentFrame();
  const a = interpolate(frame, [8, 26], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <Viewport veo={s.veo}>
      {!s.veo ? <PixelHills /> : null}
      {s.veo ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(90deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.55) 38%, transparent 62%)",
          }}
        />
      ) : null}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: s.veo ? "flex-start" : "center",
          paddingLeft: s.veo ? 70 : 0,
          gap: 70,
        }}
      >
        {!s.veo ? (
          <Img
            src={staticFile("pixel_duo.png")}
            style={{ width: 480, imageRendering: "pixelated", opacity: a }}
          />
        ) : null}
        <div style={{ fontFamily: DEV, color: INK, fontSize: 58, lineHeight: 1.45, opacity: a, maxWidth: 640 }}>
          तपाईंको <span style={{ color: CRIMSON }}>आमाको</span> लागि,
          <br />
          तपाईंको <span style={{ color: CRIMSON }}>भाइको</span> लागि।
          <br />
          <span style={{ fontSize: 38, color: MUTED }}>अंग्रेजी आवश्यक छैन।</span>
        </div>
      </div>
    </Viewport>
  );
};

// scene 2: the telephony gap — whisper fails, we don't
const Gap: React.FC<{ s: Scene }> = ({ s }) => {
  const { width } = useVideoConfig();
  return (
    <Viewport veo={s.veo}>
      {!s.veo ? <PixelHills /> : null}
      <Card
        x={width * 0.07}
        enterFrame={10}
        title="✗ whisper-large-v3"
        color={FAINT}
        lines={["99.4% WER", "हिन्दी बहाव · hallucination", "प्रयोगयोग्य छैन"]}
      />
      <Card
        x={width * 0.52}
        enterFrame={22}
        title="✓ nepaliconformer"
        color={CRIMSON}
        lines={["33.8% WER", "वास्तविक ८ kHz कल", "स्पष्ट देवनागरी"]}
      />
      <div
        style={{
          position: "absolute",
          bottom: 26,
          width: "100%",
          textAlign: "center",
          fontFamily: MONO,
          fontSize: 22,
          color: FAINT,
        }}
      >
        NepTel — real Nepali call audio · human-reviewed references
      </div>
    </Viewport>
  );
};

// scene 3: two modes
const Modes: React.FC<{ s: Scene }> = ({ s }) => {
  const { width } = useVideoConfig();
  return (
    <Viewport veo={s.veo}>
      {!s.veo ? <PixelHills /> : null}
      <Card
        x={width * 0.07}
        enterFrame={8}
        title="अफलाइन — offline"
        color={INK}
        lines={["121M params", "33.8% WER", "full-context", "उच्च शुद्धता"]}
      />
      <Card
        x={width * 0.52}
        enterFrame={20}
        title="स्ट्रिमिङ — streaming"
        color={CRIMSON}
        lines={["~520ms latency", "59.9% WER", "real-time agent", "प्रत्यक्ष कुराकानी"]}
      />
    </Viewport>
  );
};

// scene 4: leaderboard + end card
const Outro: React.FC<{ s: Scene }> = ({ s }) => {
  const frame = useCurrentFrame();
  const half = Math.floor(s.durFrames * 0.55);
  const rows = [
    { n: "1. nepaliconformer offline", v: "33.8%", ours: true },
    { n: "2. Kriti (Naamche)", v: "40.6%", ours: false },
    { n: "3. MMS-1B (Meta)", v: "81.0%", ours: false },
    { n: "4. Whisper-large-v3", v: "99.4%", ours: false },
  ];
  const endA = interpolate(frame, [half, half + 14], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <Viewport veo={s.veo}>
      {!s.veo ? <PixelHills /> : null}
      {frame < half ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              background: "#FFFFFFF2",
              border: `2px solid ${LINE}`,
              padding: "30px 44px",
              fontFamily: MONO,
            }}
          >
            <div style={{ fontSize: 26, color: CRIMSON, marginBottom: 16 }}>
              ★ NepTel LEADERBOARD ★
            </div>
            {rows.map((r, i) => {
              const ra = interpolate(frame, [8 + i * 9, 20 + i * 9], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              });
              return (
                <div
                  key={r.n}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 60,
                    fontSize: 26,
                    lineHeight: 2,
                    color: r.ours ? CRIMSON : MUTED,
                    fontWeight: r.ours ? 700 : 400,
                    opacity: ra,
                  }}
                >
                  <span>{r.n}</span>
                  <span>{r.v}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 22,
            opacity: endA,
            fontFamily: MONO,
          }}
        >
          <div style={{ fontSize: 66, fontWeight: 700, color: INK }}>
            nepali<span style={{ color: CRIMSON }}>conformer</span>
          </div>
          <div style={{ fontFamily: DEV, fontSize: 34, color: MUTED }}>
            नेपाली जसरी बोलिन्छ, त्यसरी नै चिन्नको लागि।
          </div>
          <div
            style={{
              border: `1px solid ${LINE}`,
              padding: "12px 22px",
              fontSize: 22,
              color: INK,
              background: "#F6F6F6",
            }}
          >
            $ hf download ampixa/nepali-conformer-offline · 485 MB · CPU-ready
          </div>
          <div style={{ fontSize: 26, color: CRIMSON }}>
            asr.ampixa.com · github.com/Ampixa/nepaliconformer
          </div>
          <div style={{ fontSize: 19, color: FAINT }}>ampixa labs · open weights · open benchmark</div>
        </div>
      )}
    </Viewport>
  );
};

// ------------------------------------------------------------------- root

export const Promo: React.FC<{ timeline: TimelineData }> = ({ timeline }) => {
  let at = 0;
  return (
    <AbsoluteFill style={{ background: BG }}>
      {timeline.scenes.map((s) => {
        const from = at;
        at += s.durFrames;
        return (
          <Sequence key={s.id} from={from} durationInFrames={s.durFrames}>
            {s.wav ? <Audio src={staticFile(s.wav)} /> : null}
            <AbsoluteFill>
              {s.kind === "title" ? (
                <Title dur={s.durFrames} />
              ) : s.kind === "everyday" ? (
                <Everyday s={s} />
              ) : s.kind === "gap" ? (
                <Gap s={s} />
              ) : s.kind === "modes" ? (
                <Modes s={s} />
              ) : (
                <Outro s={s} />
              )}
              {s.kind !== "title" ? <SceneASR s={s} /> : null}
            </AbsoluteFill>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

const SceneASR: React.FC<{ s: Scene }> = ({ s }) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [6, s.durFrames - 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div style={{ position: "absolute", bottom: 0, width: "100%", height: "100%" , display: "flex", flexDirection: "column", justifyContent: "flex-end"}}>
      <ASRBox transcript={s.transcript ?? ""} progress={progress} />
    </div>
  );
};
