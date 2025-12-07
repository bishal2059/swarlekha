import { Toaster } from "react-hot-toast";
import Hero from "./components/Hero";
import VoiceGenerator from "./components/VoiceGenerator";
import DemoSection from "./components/DemoSection";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="min-h-screen">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(17, 24, 39, 0.9)",
            color: "#fff",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
          },
        }}
      />
      <Hero />
      <VoiceGenerator />
      <DemoSection />
      <Footer />
    </div>
  );
}

export default App;
