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

const BG = "#0A0E12";
const PANEL = "#0D1420";
const CYAN = "#38E8FF";
const GREEN = "#41FF6A";
const RED = "#FF3B57";
const CRIMSON = "#DC143C";
const DIM = "#5A6B7A";
const MONO = "'Courier New', 'Lucida Console', Monaco, monospace";
const DEV = "'Noto Sans Devanagari', -apple-system, system-ui, sans-serif";

// ------------------------------------------------------------------ CRT bits

const Scanlines: React.FC = () => {
  const frame = useCurrentFrame();
  const flicker = 0.92 + 0.08 * Math.abs(Math.sin(frame / 1.7));
  return (
    <AbsoluteFill
      style={{
        pointerEvents: "none",
        opacity: flicker,
        background:
          "repeating-linear-gradient(0deg, rgba(0,0,0,0.22) 0px, rgba(0,0,0,0.22) 2px, transparent 2px, transparent 5px)",
      }}
    />
  );
};

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
              background: CYAN,
              boxShadow: `0 0 6px ${CYAN}66`,
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
        background: PANEL,
        borderTop: `3px solid ${CYAN}44`,
        fontFamily: MONO,
        padding: "22px 46px",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        position: "relative",
      }}
    >
      <div style={{ display: "flex", gap: 36, fontSize: 21, color: DIM }}>
        <span>
          <span
            style={{
              color: GREEN,
              textShadow: micOn ? `0 0 9px ${GREEN}` : "none",
            }}
          >
            [●]
          </span>{" "}
          MIC: <span style={{ color: GREEN }}>ACTIVE</span>
        </span>
        <span>
          MODE: <span style={{ color: CYAN }}>STREAMING (520ms)</span>
        </span>
        <span>
          LOC: <span style={{ color: CYAN }}>NEPAL</span>
        </span>
        <span style={{ marginLeft: "auto", color: DIM }}>nepaliconformer v1</span>
      </div>
      <div
        style={{
          fontFamily: DEV,
          fontSize: 38,
          lineHeight: 1.55,
          color: CYAN,
          textShadow: `0 0 10px ${CYAN}44`,
          minHeight: 120,
        }}
      >
        <span style={{ color: DIM, fontFamily: MONO }}>{">> "}</span>
        {words.slice(0, shown).join(" ")}
        {cursorOn ? <span style={{ color: GREEN }}>▌</span> : null}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 26 }}>
        <Equalizer active={progress < 0.98} />
        <div style={{ fontSize: 21, color: DIM }}>
          [{"|".repeat(Math.round(pct / 8)).padEnd(13, " ")}] {pct}%
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
  // procedural fallback backdrop: layered pixel ridgelines
  <AbsoluteFill>
    <div
      style={{
        position: "absolute",
        bottom: 0,
        width: "100%",
        height: "45%",
        background: "#101B26",
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
        background: "#16283A",
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
        background: "#0D1826EE",
        border: `2px solid ${color}`,
        boxShadow: `0 0 22px ${color}55`,
        padding: "26px 30px",
        fontFamily: MONO,
      }}
    >
      <div style={{ fontSize: 27, color, fontWeight: 700, marginBottom: 14 }}>{title}</div>
      {lines.map((l) => (
        <div key={l} style={{ fontSize: 22, color: "#C8D8E8", lineHeight: 1.8 }}>
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
          color: "#EAF4FF",
          transform: `scale(${pop})`,
          textShadow: `0 0 34px ${CYAN}33`,
        }}
      >
        nepali<span style={{ color: CRIMSON }}>conformer</span>
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 40,
          color: CYAN,
          opacity: sub,
          textShadow: `0 0 14px ${CYAN}55`,
        }}
      >
        ASR MODEL FOR NEPALI
      </div>
      <div style={{ fontFamily: DEV, fontSize: 34, color: DIM, opacity: sub }}>
        नेपालको लागि बनेको स्वचालित बोली पहिचान
      </div>
    </AbsoluteFill>
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
              "linear-gradient(90deg, rgba(6,10,14,0.82) 0%, rgba(6,10,14,0.45) 38%, transparent 62%)",
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
        <div style={{ fontFamily: DEV, color: "#EAF4FF", fontSize: 58, lineHeight: 1.45, opacity: a, maxWidth: 640, textShadow: "0 2px 18px rgba(0,0,0,0.85)" }}>
          तपाईंको <span style={{ color: CYAN }}>आमाको</span> लागि,
          <br />
          तपाईंको <span style={{ color: CYAN }}>भाइको</span> लागि।
          <br />
          <span style={{ fontSize: 38, color: DIM }}>अंग्रेजी आवश्यक छैन।</span>
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
        color={RED}
        lines={["99.4% WER", "हिन्दी बहाव · hallucination", "प्रयोगयोग्य छैन"]}
      />
      <Card
        x={width * 0.52}
        enterFrame={22}
        title="✓ nepaliconformer"
        color={CYAN}
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
          color: DIM,
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
        color={CYAN}
        lines={["121M params", "33.8% WER", "full-context", "उच्च शुद्धता"]}
      />
      <Card
        x={width * 0.52}
        enterFrame={20}
        title="स्ट्रिमिङ — streaming"
        color={GREEN}
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
              background: "#0D1826EE",
              border: `2px solid ${CYAN}`,
              boxShadow: `0 0 22px ${CYAN}44`,
              padding: "30px 44px",
              fontFamily: MONO,
            }}
          >
            <div style={{ fontSize: 26, color: CYAN, marginBottom: 16 }}>
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
                    color: r.ours ? GREEN : "#B9C9D9",
                    opacity: ra,
                    textShadow: r.ours ? `0 0 9px ${GREEN}66` : "none",
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
          <div style={{ fontSize: 66, fontWeight: 700, color: "#EAF4FF" }}>
            nepali<span style={{ color: CRIMSON }}>conformer</span>
          </div>
          <div style={{ fontFamily: DEV, fontSize: 34, color: "#B9C9D9" }}>
            नेपाली जसरी बोलिन्छ, त्यसरी नै चिन्नको लागि।
          </div>
          <div
            style={{
              border: `1px solid ${DIM}`,
              padding: "12px 22px",
              fontSize: 22,
              color: GREEN,
              background: "#000A",
            }}
          >
            $ hf download ampixa/nepali-conformer-offline · 485 MB · CPU-ready
          </div>
          <div style={{ fontSize: 26, color: CYAN }}>
            asr.ampixa.com · github.com/Ampixa/nepaliconformer
          </div>
          <div style={{ fontSize: 19, color: DIM }}>ampixa labs · open weights · open benchmark</div>
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
              <Scanlines />
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
