import { useState } from 'react'
import dynamic from 'next/dynamic'
import Head from 'next/head'

// Importa Chart.js e registra i componenti necessari
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

// Import dinamico del componente Bar (solo lato client, SSR disabilitato)
const Bar = dynamic(() => import('react-chartjs-2').then((mod) => mod.Bar), { ssr: false })

export default function Home() {
  // Stati per upload, analisi e visualizzazione
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewURL, setPreviewURL] = useState(null)
  const [emotionResult, setEmotionResult] = useState(null)
  const [probabilities, setProbabilities] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  // Gestione cambio file
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    setSelectedFile(file)
    setEmotionResult(null)
    setProbabilities(null)
    setError(null)
    setProgress(0)
    if (file) {
      setPreviewURL(URL.createObjectURL(file))
    }
  }

  // Gestione submit: simula progress bar (6 secondi) e poi mostra i risultati
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!selectedFile) {
      setError("Seleziona un'immagine!")
      return
    }
    setLoading(true)
    setProgress(0)
    
    // Simula progress bar: incrementa di 5 ogni 300ms per ~6 secondi
    let progressVal = 0
    const interval = setInterval(() => {
      progressVal += 5
      if (progressVal >= 100) {
        progressVal = 100
        clearInterval(interval)
      }
      setProgress(progressVal)
    }, 300)
    
    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      // Chiamata al backend
      const res = await fetch("http://localhost:8000/predict-emotion", {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const { error } = await res.json()
        throw new Error(error || "Errore nella predizione")
      }
      const data = await res.json()

      // Attendi i 6 secondi di progress bar prima di mostrare i risultati
      setTimeout(() => {
        setEmotionResult(data.emotion)
        setProbabilities(data.probabilities)
        setLoading(false)
      }, 6000)
    } catch (err) {
      clearInterval(interval)
      setLoading(false)
      setProgress(100)
      setError(err.message)
    }
  }

  // Configurazione del grafico
  const chartData = {
    labels: ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
    datasets: [
      {
        label: "Probabilità",
        data: probabilities || [0, 0, 0, 0, 0, 0, 0],
        backgroundColor: "rgba(59, 130, 246, 0.5)",
      },
    ],
  }

  return (
    <>
      <Head>
        <title>Emotion Detector</title>
        <meta name="description" content="Progetto Universitario - Riconoscimento Emozioni da immagini" />
      </Head>
      
      {/* Header minimal */}
      <header className="bg-white shadow-sm w-full py-4 px-6 flex items-center justify-between">
        <div className="text-xl font-bold text-gray-800">Emotion Detector</div>
        <nav className="flex space-x-6">
          <a href="#home" className="text-gray-600 hover:text-gray-900 transition">Home</a>
          <a href="#upload" className="text-gray-600 hover:text-gray-900 transition">Upload</a>
          <a href="#pricing" className="text-gray-600 hover:text-gray-900 transition">Pricing</a>
          <a href="#login" className="text-gray-600 hover:text-gray-900 transition">Login</a>
        </nav>
      </header>

      {/* Sezione "hero" con sfondo */}
      <section
        id="home"
        className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-16 md:py-24 px-4 flex items-center justify-center"
      >
        <div className="max-w-5xl text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4">Deep Learning Emotion Detection</h1>
          <p className="text-lg md:text-xl mb-8 text-white/90">
            Identifica l'emozione espressa in un'immagine con la nostra IA avanzata.
          </p>
          <a
            href="#upload"
            className="bg-white text-purple-700 font-semibold px-6 py-3 rounded-full shadow hover:bg-gray-100 transition transform hover:scale-105"
          >
            Inizia Ora
          </a>
        </div>
      </section>

      {/* Sezione di upload: simile a "Be Protected..." stile */}
      <section id="upload" className="w-full bg-white py-16 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between">
          {/* Colonna Sinistra: testo / bullet points */}
          <div className="md:w-1/2 mb-10 md:mb-0 md:pr-8">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4 leading-tight">
              Be Protected Against Emotional Misinterpretations
            </h2>
            <p className="text-gray-600 text-lg mb-6">
              Carica un'immagine e verifica in tempo reale quale emozione viene espressa.
              La nostra tecnologia sfrutta reti neurali e analisi facciale avanzata.
            </p>
            <ul className="list-disc list-inside text-gray-600 mb-6 space-y-2">
              <li>Supporto a immagini di varie risoluzioni</li>
              <li>Interfaccia semplice e veloce</li>
              <li>Nessun costo nascosto: trasparenza garantita</li>
            </ul>
            <p className="text-gray-500 italic">
              Un semplice upload e la nostra IA farà il resto.
            </p>
          </div>

          {/* Colonna Destra: box upload */}
          <div className="md:w-1/2 bg-gray-50 p-6 rounded-lg shadow-lg">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Carica Immagine</h3>
            
            <form onSubmit={handleSubmit} className="flex flex-col items-start space-y-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="border border-gray-300 p-2 rounded w-full focus:outline-none focus:ring-2 focus:ring-purple-600"
              />
              <button
                type="submit"
                className="bg-purple-600 text-white px-6 py-3 rounded w-full text-center font-semibold hover:bg-purple-700 transition transform hover:scale-105"
              >
                Analizza Ora
              </button>
            </form>

            {loading && (
              <div className="mt-6 w-full">
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-purple-600 h-4 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-center mt-2 text-gray-600">Analisi in corso... {progress}%</p>
              </div>
            )}

            {error && <p className="text-red-500 mt-4">{error}</p>}

            {/* Se l'utente ha caricato un file e abbiamo un'anteprima */}
            {previewURL && !loading && (
              <div className="mt-8 flex flex-col md:flex-row md:space-x-6 items-center">
                {/* Anteprima immagine */}
                <div className="mb-6 md:mb-0">
                  <h4 className="text-lg font-semibold mb-2 text-gray-800">Anteprima</h4>
                  <img
                    src={previewURL}
                    alt="Anteprima"
                    className="w-64 h-64 object-cover rounded shadow-lg transform transition duration-300 hover:scale-105"
                  />
                </div>
                {/* Risultato e grafico */}
                {emotionResult && probabilities && (
                  <div className="bg-white p-4 rounded shadow-lg">
                    <h4 className="text-lg font-semibold mb-2 text-gray-800">Risultato</h4>
                    <p className="mb-4 text-gray-700">Emozione: <strong>{emotionResult}</strong></p>
                    <div className="w-64 h-64">
                      <Bar
                        data={{
                          labels: ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
                          datasets: [
                            {
                              label: "Probabilità",
                              data: probabilities,
                              backgroundColor: "rgba(59, 130, 246, 0.5)",
                            },
                          ],
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                            legend: { position: "top" },
                            title: { display: true, text: "Distribuzione delle Probabilità" },
                          },
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Footer minimal */}
      <footer className="w-full bg-white border-t border-gray-200 py-6 px-4 flex items-center justify-between">
        <p className="text-sm text-gray-500">
          © 2024/25 Emotion Detector - Biometric System Project.
        </p>
        <div className="flex space-x-4 text-sm">
          <a href="#privacy" className="text-gray-500 hover:text-gray-700 transition">Privacy</a>
          <a href="#terms" className="text-gray-500 hover:text-gray-700 transition">Terms</a>
          <a href="#contact" className="text-gray-500 hover:text-gray-700 transition">Contact</a>
        </div>
      </footer>

      {/* Spinner CSS */}
      <style jsx>{`
        .loader {
          border-top-color: #3490dc;
          animation: spinner 0.6s linear infinite;
        }
        @keyframes spinner {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  )
}
