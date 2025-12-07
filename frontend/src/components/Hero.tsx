import { motion } from "framer-motion";
import { Mic2, Sparkles, Zap } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden px-4">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float"></div>
        <div
          className="absolute top-40 right-10 w-72 h-72 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute -bottom-8 left-20 w-72 h-72 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-float"
          style={{ animationDelay: "4s" }}
        ></div>
      </div>

      <div className="relative z-10 max-w-6xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="flex justify-center mb-6">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="relative"
            >
              <div className="w-24 h-24 bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500 rounded-full flex items-center justify-center">
                <Mic2 className="w-12 h-12 text-white" />
              </div>
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500 rounded-full blur-xl opacity-50 animate-pulse-slow"></div>
            </motion.div>
          </div>

          <h1 className="text-6xl md:text-8xl font-bold mb-6 gradient-text">
            Swarlekha TTS
          </h1>

          <p className="text-xl md:text-2xl text-gray-300 mb-4 max-w-3xl mx-auto">
            Transform text into natural-sounding speech with{" "}
            <span className="text-purple-400 font-semibold">
              advanced voice cloning
            </span>
          </p>

          <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto">
            Powered by state-of-the-art AI technology for realistic voice
            generation
          </p>

          <div className="flex flex-wrap justify-center gap-6 mb-12">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-3 px-6 py-3 glass-morphism rounded-full"
            >
              <Sparkles className="w-5 h-5 text-yellow-400" />
              <span>Voice Cloning</span>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-3 px-6 py-3 glass-morphism rounded-full"
            >
              <Zap className="w-5 h-5 text-cyan-400" />
              <span>Fast Generation</span>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-3 px-6 py-3 glass-morphism rounded-full"
            >
              <Mic2 className="w-5 h-5 text-blue-400" />
              <span>Natural Sound</span>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex justify-center"
          >
            <a
              href="#generator"
              className="btn-primary inline-flex items-center gap-2"
            >
              Get Started
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </a>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
