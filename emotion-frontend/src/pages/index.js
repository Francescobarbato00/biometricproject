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
  // Stati per la sezione di upload e predizione
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

    // Inizia la simulazione della progress bar: incrementa di 5 ogni 300ms (20 * 300ms = 6000ms)
    let progressVal = 0
    const interval = setInterval(() => {
      progressVal += 5
      if (progressVal >= 100) {
        progressVal = 100
        clearInterval(interval)
      }
      setProgress(progressVal)
    }, 300)

    try {
      // Esegui la richiesta al backend
      const res = await fetch("http://localhost:8000/predict-emotion", {
        method: "POST",
        body: (() => {
          const fd = new FormData()
          fd.append("file", selectedFile)
          return fd
        })(),
      })
      if (!res.ok) {
        const { error } = await res.json()
        throw new Error(error || "Errore nella predizione")
      }
      const data = await res.json()
      
      // Forza una pausa fino a che la progress bar raggiunge il 100%
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
        <title>Riconoscimento Emozioni</title>
        <meta name="description" content="Progetto Universitario - Riconoscimento Emozioni da immagini" />
      </Head>
      <div className="min-h-screen flex flex-col font-jost">
        {/* Header */}
        <header className="bg-blue-600 text-white py-4 shadow-lg fixed top-0 left-0 right-0 z-50">
          <div className="container mx-auto px-4 flex items-center justify-between">
            <div className="text-2xl font-bold">EmotionProject</div>
            <nav className="flex space-x-6">
              <a href="#hero" className="hover:text-blue-300 transition duration-300">Home</a>
              <a href="#upload" className="hover:text-blue-300 transition duration-300">Upload</a>
              <a href="#about" className="hover:text-blue-300 transition duration-300">About</a>
              <a href="#contact" className="hover:text-blue-300 transition duration-300">Contact</a>
            </nav>
          </div>
        </header>

        {/* Hero Section */}
        <section id="hero" className="w-full bg-gray-900 text-white py-20 px-4 mt-16">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between">
            <div className="md:w-1/2 mb-8 md:mb-0 md:pr-8">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
                Riconoscimento delle Emozioni
              </h1>
              <p className="text-lg md:text-xl text-gray-300 mb-6">
                Analizza rapidamente un'immagine per scoprire quale emozione è espressa.
                Sfruttiamo modelli di Deep Learning per garantire risultati affidabili e in tempo reale.
              </p>
              <a
                href="#upload"
                className="inline-block bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-full font-semibold transition transform hover:scale-105"
              >
                Inizia Ora
              </a>
            </div>
            <div className="md:w-1/2 flex items-center justify-center">
              <img
                src="/hero_emotion.jpg"
                alt="Emotion Recognition"
                className="w-full max-w-md rounded-lg shadow-lg transform hover:scale-105 transition duration-300"
              />
            </div>
          </div>
        </section>

        {/* Upload Section */}
        <section id="upload" className="w-full bg-white py-16 px-4">
          <div className="max-w-7xl mx-auto flex flex-col items-center">
            <h2 className="text-4xl font-bold mb-6 text-gray-800">Carica la Tua Immagine</h2>
            <form onSubmit={handleSubmit} className="w-full max-w-lg flex flex-col items-center space-y-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full border border-gray-300 p-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-600"
              />
              <button
                type="submit"
                className="w-full bg-blue-600 text-white px-6 py-3 rounded transform transition duration-300 hover:scale-105 hover:bg-blue-700"
              >
                Carica e Analizza
              </button>
            </form>

            {loading && (
              <div className="w-full max-w-lg mt-6">
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="text-center mt-2 text-gray-600">Analisi in corso... {progress}%</p>
              </div>
            )}

            {error && <p className="text-red-500 mt-4">{error}</p>}

            {previewURL && (
              <div className="mt-8 flex flex-col md:flex-row md:space-x-8 items-center">
                {/* Anteprima immagine */}
                <div className="mb-6 md:mb-0">
                  <h3 className="text-2xl font-semibold mb-2 text-gray-800">Anteprima Immagine</h3>
                  <img
                    src={previewURL}
                    alt="Anteprima"
                    className="w-64 h-64 object-cover rounded shadow-lg transform transition duration-300 hover:scale-105"
                  />
                </div>
                {/* Risultato e grafico */}
                {emotionResult && probabilities && (
                  <div className="bg-white p-6 rounded shadow-lg">
                    <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                      Emozione Predetta: {emotionResult}
                    </h3>
                    <div className="w-64 h-64">
                      <Bar
                        data={chartData}
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
        </section>

        {/* Footer */}
        <footer id="contact" className="bg-blue-600 text-white py-6">
          <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between">
            <div className="mb-4 md:mb-0">
              <h3 className="text-xl font-bold">EmotionProject</h3>
              <p className="text-sm text-blue-200">Riconoscimento Facciale &copy; 2023</p>
            </div>
            <div className="flex space-x-4">
              <a href="#privacy" className="text-blue-200 hover:text-white transition">Privacy</a>
              <a href="#terms" className="text-blue-200 hover:text-white transition">Terms</a>
              <a href="#contact" className="text-blue-200 hover:text-white transition">Contatti</a>
            </div>
          </div>
        </footer>
      </div>

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
