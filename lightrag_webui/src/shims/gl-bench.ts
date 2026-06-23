// No-op stand-in for gl-bench, a GPU FPS-overlay dev tool pulled in transitively
// by @cosmos.gl/graph. Its UMD bundle exposes no default export that rolldown can
// synthesise, which breaks the production build; the overlay is non-essential and
// off by default. The methods are no-ops — JS ignores any arguments cosmos passes.
export default class GLBench {
  begin(): void {}
  end(): void {}
  nextFrame(): void {}
  dispose(): void {}
}
