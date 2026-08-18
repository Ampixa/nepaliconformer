import { Composition, staticFile } from "remotion";
import { Promo, TimelineData } from "./Promo";

const FPS = 30;

const calc = async () => {
  const res = await fetch(staticFile("timeline.json"));
  const timeline = (await res.json()) as TimelineData;
  const total = timeline.scenes.reduce((a, s) => a + s.durFrames, 0);
  return { durationInFrames: total, props: { timeline } };
};

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="Promo"
        component={Promo}
        width={1920}
        height={1080}
        fps={FPS}
        defaultProps={{ timeline: { scenes: [] } as TimelineData }}
        calculateMetadata={async () => {
          const { durationInFrames, props } = await calc();
          return { durationInFrames, props };
        }}
      />
      <Composition
        id="PromoVertical"
        component={Promo}
        width={1080}
        height={1920}
        fps={FPS}
        defaultProps={{ timeline: { scenes: [] } as TimelineData }}
        calculateMetadata={async () => {
          const { durationInFrames, props } = await calc();
          return { durationInFrames, props };
        }}
      />
    </>
  );
};
