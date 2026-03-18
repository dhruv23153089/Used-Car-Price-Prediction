import "./App.css";
import CarForm from "./components/CarForm";
import heroCar from "./assets/hero-car1.png";

function App() {
  return (
    <main className="app-shell">
      <div className="background-orb background-orb-left" aria-hidden="true" />
      <div className="background-orb background-orb-right" aria-hidden="true" />

      <section className="hero-card">
        <div className="hero-content">
          <p className="eyebrow">AI Powered Valuation</p>
          <h1>Used Car Price Predictor</h1>
          <p className="hero-subtitle">
            Get an instant market-backed estimate using your car details. Fast, clean, and built for confident pricing decisions.
          </p>

          <div className="trust-row" role="list" aria-label="Highlights">
            <span role="listitem">Data-driven estimate</span>
            <span role="listitem">Real-time prediction</span>
            <span role="listitem">Simple 8-field input</span>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="hero-visual-glow hero-visual-glow-warm" />
          <div className="hero-visual-glow hero-visual-glow-cool" />
          <div className="hero-motion-scene">
            <div className="hero-speed-aura" />
            <div className="hero-speed-trails" />
            <div className="hero-car-stage">
              <img className="hero-car-image" src={heroCar} alt="" />
            </div>
          </div>
        </div>
      </section>

      <section className="predictor-panel">
        <div className="panel-header">
          <h2>Enter Vehicle Details</h2>
          <p>Fill in the details below to generate your estimated resale price.</p>
        </div>

        <CarForm />
      </section>
    </main>
  );
}

export default App;
