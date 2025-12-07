# Swarlekha TTS Frontend

Modern React TypeScript frontend for Swarlekha Text-to-Speech with voice cloning.

## Features

- 🎨 Beautiful, modern UI with glassmorphism design
- 🎤 Voice cloning with audio upload or recording
- 📝 Text-to-speech generation
- 🎵 Real-time audio playback and download
- 📱 Fully responsive design
- ⚡ Built with Vite for fast development

## Tech Stack

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Framer Motion (animations)
- Axios (API calls)
- React Dropzone (file uploads)
- React Hot Toast (notifications)
- Lucide React (icons)

## Installation

```bash
# Install dependencies
npm install
```

## Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:8000
```

## Running the Application

```bash
# Development mode with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint the code
npm run lint
```

The application will start at `http://localhost:3000`

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Hero.tsx        # Landing hero section
│   │   ├── VoiceGenerator.tsx  # Main voice generation interface
│   │   ├── DemoSection.tsx # Demo audio examples
│   │   └── Footer.tsx      # Footer component
│   ├── services/           # API services
│   │   └── api.ts         # API client
│   ├── App.tsx            # Main app component
│   ├── main.tsx           # Entry point
│   ├── index.css          # Global styles
│   └── vite-env.d.ts     # TypeScript definitions
├── public/                # Static assets
├── index.html            # HTML template
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
├── vite.config.ts        # Vite config
└── tailwind.config.js    # Tailwind CSS config
```

## Features in Detail

### Voice Generation

- Enter text up to 5000 characters
- Choose between default voice or voice cloning
- Upload reference audio or record directly
- Real-time generation with loading states

### Audio Upload

- Drag & drop or click to browse
- Supports WAV, MP3, M4A, OGG formats
- Visual feedback for uploaded files

### Voice Recording

- Record audio directly from browser
- Start/stop recording controls
- Automatic save and processing

### Demo Section

- Pre-configured demo examples
- Sample audio playback
- Default vs cloned voice comparison

## API Integration

The frontend connects to the FastAPI backend:

- `GET /api/health` - Health check
- `POST /api/generate` - Generate voice
- `GET /api/voices` - List generated voices

## Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```javascript
colors: {
  primary: { /* your colors */ },
  accent: { /* your colors */ }
}
```

### Animations

Modify Framer Motion settings in components for different animations.

### API URL

Change the API URL in `.env` file to point to your backend server.

## Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

## Deployment

### Static Hosting (Netlify, Vercel, etc.)

1. Build the project: `npm run build`
2. Deploy the `dist/` directory
3. Set environment variables in your hosting platform

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

Same as the main Swarlekha project
