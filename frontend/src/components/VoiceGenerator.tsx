import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Mic, Play, Download, Loader, X, Volume2 } from "lucide-react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { generateVoice } from "../services/api";

const VoiceGenerator = () => {
  const [text, setText] = useState("");
  const [voiceName, setVoiceName] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [useDefault, setUseDefault] = useState(true);

  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setAudioFile(acceptedFiles[0]);
      setUseDefault(false);
      toast.success("Audio file uploaded!");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".wav", ".mp3", ".m4a", ".ogg"],
    },
    maxFiles: 1,
  });

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        const audioFile = new File([audioBlob], "recorded_audio.wav", {
          type: "audio/wav",
        });
        setAudioFile(audioFile);
        setUseDefault(false);
        stream.getTracks().forEach((track) => track.stop());
        toast.success("Recording saved!");
      };

      mediaRecorder.start();
      setIsRecording(true);
      toast.success("Recording started...");
    } catch (error) {
      toast.error("Failed to access microphone");
      console.error(error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text");
      return;
    }

    setIsGenerating(true);
    try {
      const audioBlob = await generateVoice(
        text,
        useDefault ? null : audioFile,
        voiceName || "generated"
      );

      const audioUrl = URL.createObjectURL(audioBlob);
      setGeneratedAudio(audioUrl);
      toast.success("Voice generated successfully!");
    } catch (error) {
      toast.error("Failed to generate voice");
      console.error(error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (generatedAudio) {
      const a = document.createElement("a");
      a.href = generatedAudio;
      a.download = `${voiceName || "generated"}_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      toast.success("Audio downloaded!");
    }
  };

  const playAudio = () => {
    if (audioRef.current) {
      audioRef.current.play();
    }
  };

  return (
    <section id="generator" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 gradient-text">
            Generate Your Voice
          </h2>
          <p className="text-center text-gray-400 mb-12 max-w-2xl mx-auto">
            Enter your text, upload a reference audio for voice cloning, or use
            the default voice
          </p>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Left Column - Input Section */}
            <div className="space-y-6">
              {/* Text Input */}
              <div className="card">
                <label className="block text-sm font-medium mb-2">
                  Text to Speech
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter the text you want to convert to speech..."
                  className="input-field min-h-[200px] resize-none"
                  maxLength={5000}
                />
                <div className="mt-2 text-sm text-gray-400 text-right">
                  {text.length} / 5000
                </div>
              </div>

              {/* Voice Name */}
              <div className="card">
                <label className="block text-sm font-medium mb-2">
                  Voice Name (Optional)
                </label>
                <input
                  type="text"
                  value={voiceName}
                  onChange={(e) => setVoiceName(e.target.value)}
                  placeholder="e.g., my_voice"
                  className="input-field"
                />
              </div>

              {/* Voice Selection */}
              <div className="card">
                <label className="block text-sm font-medium mb-4">
                  Voice Selection
                </label>

                <div className="flex gap-4 mb-4">
                  <button
                    onClick={() => {
                      setUseDefault(true);
                      setAudioFile(null);
                    }}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      useDefault
                        ? "bg-gradient-to-r from-blue-500 to-purple-600"
                        : "glass-morphism hover:bg-white/20"
                    }`}
                  >
                    Default Voice
                  </button>
                  <button
                    onClick={() => setUseDefault(false)}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      !useDefault
                        ? "bg-gradient-to-r from-blue-500 to-purple-600"
                        : "glass-morphism hover:bg-white/20"
                    }`}
                  >
                    Clone Voice
                  </button>
                </div>

                <AnimatePresence>
                  {!useDefault && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-4"
                    >
                      {/* Upload/Drop Zone */}
                      <div
                        {...getRootProps()}
                        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
                          isDragActive
                            ? "border-purple-500 bg-purple-500/10"
                            : "border-gray-600 hover:border-purple-500"
                        }`}
                      >
                        <input {...getInputProps()} />
                        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                        {audioFile ? (
                          <div className="flex items-center justify-center gap-2">
                            <Volume2 className="w-5 h-5 text-green-400" />
                            <span className="text-green-400">
                              {audioFile.name}
                            </span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setAudioFile(null);
                              }}
                              className="ml-2 text-red-400 hover:text-red-300"
                            >
                              <X className="w-5 h-5" />
                            </button>
                          </div>
                        ) : (
                          <>
                            <p className="text-gray-300 mb-2">
                              {isDragActive
                                ? "Drop your audio file here"
                                : "Drag & drop an audio file"}
                            </p>
                            <p className="text-sm text-gray-500">
                              or click to browse (WAV, MP3, M4A, OGG)
                            </p>
                          </>
                        )}
                      </div>

                      {/* Record Button */}
                      <div className="text-center">
                        <div className="text-sm text-gray-400 mb-3">
                          Or record your voice
                        </div>
                        <button
                          onClick={isRecording ? stopRecording : startRecording}
                          className={`btn-primary inline-flex items-center gap-2 ${
                            isRecording ? "animate-pulse" : ""
                          }`}
                        >
                          <Mic className="w-5 h-5" />
                          {isRecording ? "Stop Recording" : "Record Audio"}
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !text.trim()}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 py-4"
              >
                {isGenerating ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Generate Voice
                  </>
                )}
              </button>
            </div>

            {/* Right Column - Output Section */}
            <div className="space-y-6">
              <div className="card h-full flex flex-col">
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Volume2 className="w-6 h-6 text-purple-400" />
                  Generated Audio
                </h3>

                <AnimatePresence mode="wait">
                  {generatedAudio ? (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="flex-1 flex flex-col items-center justify-center space-y-6"
                    >
                      {/* Audio Visualizer Placeholder */}
                      <div className="w-full h-40 glass-morphism rounded-lg flex items-center justify-center">
                        <div className="flex items-end gap-1 h-20">
                          {[...Array(20)].map((_, i) => (
                            <motion.div
                              key={i}
                              className="w-2 bg-gradient-to-t from-blue-500 to-purple-500 rounded-full"
                              animate={{
                                height: [20, 60, 20],
                              }}
                              transition={{
                                duration: 0.8,
                                repeat: Infinity,
                                delay: i * 0.1,
                              }}
                            />
                          ))}
                        </div>
                      </div>

                      <audio
                        ref={audioRef}
                        src={generatedAudio}
                        className="w-full"
                        controls
                      />

                      <div className="flex gap-4 w-full">
                        <button
                          onClick={playAudio}
                          className="flex-1 btn-secondary flex items-center justify-center gap-2"
                        >
                          <Play className="w-5 h-5" />
                          Play
                        </button>
                        <button
                          onClick={handleDownload}
                          className="flex-1 btn-primary flex items-center justify-center gap-2"
                        >
                          <Download className="w-5 h-5" />
                          Download
                        </button>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-gray-400"
                    >
                      <Volume2 className="w-20 h-20 mb-4 opacity-20" />
                      <p className="text-lg">
                        Your generated audio will appear here
                      </p>
                      <p className="text-sm mt-2">
                        Enter text and click Generate to start
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default VoiceGenerator;
