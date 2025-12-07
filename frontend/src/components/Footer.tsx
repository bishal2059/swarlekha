import { Github, Twitter, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 px-4 border-t border-white/10">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div>
            <h3 className="text-2xl font-bold gradient-text mb-4">
              Swarlekha TTS
            </h3>
            <p className="text-gray-400 text-sm">
              Advanced Text-to-Speech with state-of-the-art voice cloning
              technology
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>
                <a
                  href="#generator"
                  className="hover:text-purple-400 transition-colors"
                >
                  Voice Generator
                </a>
              </li>
              <li>
                <a
                  href="#demos"
                  className="hover:text-purple-400 transition-colors"
                >
                  Demos
                </a>
              </li>
              <li>
                <a
                  href="/api/health"
                  className="hover:text-purple-400 transition-colors"
                >
                  API Health
                </a>
              </li>
            </ul>
          </div>

          {/* Social */}
          <div>
            <h4 className="font-semibold mb-4">Connect</h4>
            <div className="flex gap-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 glass-morphism rounded-full flex items-center justify-center hover:bg-white/20 transition-all"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 glass-morphism rounded-full flex items-center justify-center hover:bg-white/20 transition-all"
              >
                <Twitter className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="pt-8 border-t border-white/10 text-center">
          <p className="text-gray-400 text-sm flex items-center justify-center gap-2">
            Made with <Heart className="w-4 h-4 text-red-400" /> using Swarlekha
            TTS Model
          </p>
          <p className="text-gray-500 text-xs mt-2">
            © {new Date().getFullYear()} Swarlekha TTS. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
