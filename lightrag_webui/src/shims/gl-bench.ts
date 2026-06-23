// No-op stand-in for gl-bench, a GPU FPS-overlay dev tool pulled in transitively
// by @cosmos.gl/graph. Its UMD bundle exposes no default export that rolldown can
// synthesise, which breaks the production build; the overlay is non-essential and
// off by default, so a no-op class with the methods cosmos may call is sufficient.
export default class GLBench {
  constructor(..._args: unknown[]) {}
  begin(..._args: unknown[]): void {}
  end(..._args: unknown[]): void {}
  nextFrame(..._args: unknown[]): void {}
  dispose(): void {}
}
