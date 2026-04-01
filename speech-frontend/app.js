const ANALYSIS_SAMPLE_RATE = 16000;
const FRAME_SIZE = 400;
const HOP_SIZE = 160;
const FFT_SIZE = 512;
const MEL_BINS = 40;
const PRE_EMPHASIS = 0.97;
const SPECTROGRAM_DYNAMIC_RANGE_DB = 70;
const MAX_DISPLAY_FREQUENCY_HZ = 5000;

const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const frameSlider = document.querySelector("#frameSlider");

const statusText = document.querySelector("#statusText");
const inputRateText = document.querySelector("#inputRateText");
const analysisRateText = document.querySelector("#analysisRateText");
const durationText = document.querySelector("#durationText");
const frameCountText = document.querySelector("#frameCountText");
const melBinsText = document.querySelector("#melBinsText");
const frameIndexText = document.querySelector("#frameIndexText");

const waveformCanvas = document.querySelector("#waveformCanvas");
const spectrogramCanvas = document.querySelector("#spectrogramCanvas");
const frameCanvas = document.querySelector("#frameCanvas");
const melCanvas = document.querySelector("#melCanvas");

const waveformCtx = waveformCanvas.getContext("2d");
const spectrogramCtx = spectrogramCanvas.getContext("2d");
const frameCtx = frameCanvas.getContext("2d");
const melCtx = melCanvas.getContext("2d");

analysisRateText.textContent = `${ANALYSIS_SAMPLE_RATE.toLocaleString()} Hz`;
melBinsText.textContent = `${MEL_BINS}`;

let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let analyserNode;
let animationFrameId;
let rawChunks = [];
let recordedSamples = null;
let latestAnalysis = null;
let isRecording = false;

const hammingWindow = createHammingWindow(FRAME_SIZE);
const melFilterBank = createMelFilterBank({
  sampleRate: ANALYSIS_SAMPLE_RATE,
  fftSize: FFT_SIZE,
  melBins: MEL_BINS,
  minHz: 20,
  maxHz: ANALYSIS_SAMPLE_RATE / 2,
});

drawEmptyState(waveformCtx, waveformCanvas, "Press Start recording to capture audio.");
drawEmptyState(spectrogramCtx, spectrogramCanvas, "Spectrogram will appear after analysis.");
drawEmptyState(frameCtx, frameCanvas, "Select a frame after analysis.");
drawEmptyState(melCtx, melCanvas, "The same frame after log-mel will appear here.");

startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
frameSlider.addEventListener("input", updateSelectedFrame);

async function startRecording() {
  if (isRecording) {
    return;
  }

  try {
    statusText.textContent = "Requesting microphone access";
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    await audioContext.resume();

    rawChunks = [];
    recordedSamples = null;
    latestAnalysis = null;
    frameSlider.disabled = true;
    durationText.textContent = "-";
    frameCountText.textContent = "-";

    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    analyserNode.smoothingTimeConstant = 0.85;

    processorNode = audioContext.createScriptProcessor(2048, 1, 1);
    processorNode.onaudioprocess = (event) => {
      if (!isRecording) {
        return;
      }

      const input = event.inputBuffer.getChannelData(0);
      rawChunks.push(new Float32Array(input));
    };

    sourceNode.connect(analyserNode);
    analyserNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    inputRateText.textContent = `${audioContext.sampleRate.toLocaleString()} Hz`;
    statusText.textContent = "Recording";
    isRecording = true;
    startButton.disabled = true;
    stopButton.disabled = false;

    renderLiveWaveform();
  } catch (error) {
    console.error(error);
    statusText.textContent = "Microphone access failed";
  }
}

function stopRecording() {
  if (!isRecording) {
    return;
  }

  isRecording = false;
  startButton.disabled = false;
  stopButton.disabled = true;

  cancelAnimationFrame(animationFrameId);

  processorNode?.disconnect();
  analyserNode?.disconnect();
  sourceNode?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();

  processorNode = null;
  analyserNode = null;
  sourceNode = null;
  mediaStream = null;
  audioContext = null;

  recordedSamples = concatenateChunks(rawChunks);
  durationText.textContent = formatSeconds(recordedSamples.length / inputRateFromLabel());
  statusText.textContent =
    recordedSamples.length > 0 ? "Recording captured" : "No audio captured";

  if (recordedSamples.length > 0) {
    drawWaveform(waveformCtx, waveformCanvas, normalizeToCanvas(recordedSamples, waveformCanvas.width));
    runAnalysis();
  } else {
    drawEmptyState(waveformCtx, waveformCanvas, "No waveform captured.");
  }
}

function runAnalysis() {
  if (!recordedSamples || recordedSamples.length === 0) {
    return;
  }

  const resampled = downsampleToTargetRate(recordedSamples, inputRateFromLabel(), ANALYSIS_SAMPLE_RATE);
  const normalized = normalizeSignal(resampled);
  const emphasized = preEmphasize(normalized, PRE_EMPHASIS);
  const frames = frameSignal(emphasized, FRAME_SIZE, HOP_SIZE);
  const windowedFrames = frames.map((frame) => multiply(frame, hammingWindow));
  const stft = windowedFrames.map((frame) => powerSpectrum(frame, FFT_SIZE));
  const spectrogram = stft.map((spectrum) => powerToDecibels(spectrum));
  const logMels = stft.map((spectrum) => applyMelFilters(spectrum, melFilterBank));
  const clippedSpectrogram = clampSpectrogramDynamicRange(
    spectrogram,
    SPECTROGRAM_DYNAMIC_RANGE_DB,
  );
  const spectrogramRange = getFiniteRange(clippedSpectrogram);

  latestAnalysis = {
    resampled: emphasized,
    frames,
    stft,
    spectrogram: clippedSpectrogram,
    logMels,
  };

  frameCountText.textContent = `${frames.length}`;
  durationText.textContent = formatSeconds(emphasized.length / ANALYSIS_SAMPLE_RATE);
  statusText.textContent = Number.isFinite(spectrogramRange.min) && Number.isFinite(spectrogramRange.max)
    ? `Analysis complete (${frames.length} frames, ${stft[0]?.length ?? 0} bins, ${spectrogramRange.min.toFixed(1)} to ${spectrogramRange.max.toFixed(1)} log power)`
    : "Analysis complete";

  frameSlider.disabled = frames.length === 0;
  frameSlider.min = "0";
  frameSlider.max = `${Math.max(0, frames.length - 1)}`;
  frameSlider.value = "0";
  frameIndexText.textContent = "0";

  drawSpectrogram(spectrogramCtx, spectrogramCanvas, clippedSpectrogram, 0);
  updateSelectedFrame();
}

function updateSelectedFrame() {
  if (!latestAnalysis || latestAnalysis.stft.length === 0) {
    return;
  }

  const index = Number(frameSlider.value);
  const frame = latestAnalysis.stft[index];
  const logMelFrame = latestAnalysis.logMels[index];
  frameIndexText.textContent = `${index}`;
  drawSpectrogram(spectrogramCtx, spectrogramCanvas, latestAnalysis.spectrogram, index);
  drawVectorHeatmap(frameCtx, frameCanvas, frame, "spectrum");
  drawVectorHeatmap(melCtx, melCanvas, logMelFrame, "mel");
}

function renderLiveWaveform() {
  if (!isRecording || !analyserNode) {
    return;
  }

  const data = new Uint8Array(analyserNode.fftSize);
  analyserNode.getByteTimeDomainData(data);
  const normalized = Array.from(data, (value) => value / 128 - 1);
  drawWaveform(waveformCtx, waveformCanvas, normalized);
  animationFrameId = requestAnimationFrame(renderLiveWaveform);
}

function inputRateFromLabel() {
  const label = inputRateText.textContent || "";
  const numeric = Number(label.replace(/[^0-9.]/g, ""));
  return numeric || ANALYSIS_SAMPLE_RATE;
}

function concatenateChunks(chunks) {
  if (chunks.length === 0) {
    return new Float32Array();
  }

  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function downsampleToTargetRate(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) {
    return samples;
  }

  const ratio = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.floor(samples.length / ratio));
  const result = new Float32Array(outputLength);

  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(samples.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;

    for (let cursor = start; cursor < end; cursor += 1) {
      sum += samples[cursor];
      count += 1;
    }

    result[index] = count > 0 ? sum / count : 0;
  }

  return result;
}

function frameSignal(samples, frameSize, hopSize) {
  if (samples.length === 0) {
    return [];
  }

  const frames = [];

  for (let start = 0; start < samples.length; start += hopSize) {
    const frame = new Float32Array(frameSize);
    frame.set(samples.subarray(start, Math.min(start + frameSize, samples.length)));
    frames.push(frame);

    if (start + frameSize >= samples.length) {
      break;
    }
  }

  return frames;
}

function normalizeSignal(samples) {
  let peak = 0;

  for (const sample of samples) {
    peak = Math.max(peak, Math.abs(sample));
  }

  if (peak < 1e-8) {
    return samples;
  }

  const normalized = new Float32Array(samples.length);

  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = samples[index] / peak;
  }

  return normalized;
}

function preEmphasize(samples, coefficient) {
  if (samples.length === 0) {
    return samples;
  }

  const emphasized = new Float32Array(samples.length);
  emphasized[0] = samples[0];

  for (let index = 1; index < samples.length; index += 1) {
    emphasized[index] = samples[index] - coefficient * samples[index - 1];
  }

  return emphasized;
}

function createHammingWindow(length) {
  const window = new Float32Array(length);

  for (let index = 0; index < length; index += 1) {
    window[index] = 0.54 - 0.46 * Math.cos((2 * Math.PI * index) / (length - 1));
  }

  return window;
}

function multiply(signal, window) {
  const result = new Float32Array(signal.length);

  for (let index = 0; index < signal.length; index += 1) {
    result[index] = signal[index] * window[index];
  }

  return result;
}

function powerSpectrum(frame, fftSize) {
  const bins = fftSize / 2 + 1;
  const spectrum = new Float32Array(bins);
  const padded = new Float32Array(fftSize);
  padded.set(frame.subarray(0, Math.min(frame.length, fftSize)));

  for (let bin = 0; bin < bins; bin += 1) {
    let real = 0;
    let imag = 0;

    for (let sampleIndex = 0; sampleIndex < fftSize; sampleIndex += 1) {
      const angle = (2 * Math.PI * bin * sampleIndex) / fftSize;
      const sample = padded[sampleIndex];
      real += sample * Math.cos(angle);
      imag -= sample * Math.sin(angle);
    }

    spectrum[bin] = real ** 2 + imag ** 2;
  }

  return spectrum;
}

function createMelFilterBank({ sampleRate, fftSize, melBins, minHz, maxHz }) {
  const lowMel = hzToMel(minHz);
  const highMel = hzToMel(maxHz);
  const melPoints = new Float32Array(melBins + 2);

  for (let index = 0; index < melPoints.length; index += 1) {
    melPoints[index] = lowMel + ((highMel - lowMel) * index) / (melBins + 1);
  }

  const hzPoints = Array.from(melPoints, melToHz);
  const maxBin = fftSize / 2;
  const binPoints = hzPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));
  const filterBank = [];

  for (let melIndex = 1; melIndex <= melBins; melIndex += 1) {
    const filter = new Float32Array(maxBin + 1);
    const left = Math.max(0, Math.min(maxBin, binPoints[melIndex - 1]));
    const center = Math.max(left + 1, Math.min(maxBin, binPoints[melIndex]));
    const right = Math.max(center + 1, Math.min(maxBin, binPoints[melIndex + 1]));

    for (let bin = left; bin < center; bin += 1) {
      filter[bin] = (bin - left) / (center - left);
    }

    for (let bin = center; bin < right; bin += 1) {
      filter[bin] = (right - bin) / (right - center);
    }

    filterBank.push(filter);
  }

  return filterBank;
}

function applyMelFilters(spectrum, filterBank) {
  return filterBank.map((filter) => {
    let energy = 0;

    for (let index = 0; index < spectrum.length; index += 1) {
      energy += spectrum[index] * filter[index];
    }

    return Math.log10(energy + 1e-10);
  });
}

function powerToDecibels(spectrum) {
  return Array.from(spectrum, (value) => 10 * Math.log10(value + 1e-10));
}

function clampSpectrogramDynamicRange(spectrogram, dynamicRangeDb) {
  const { max } = getFiniteRange(spectrogram);

  if (!Number.isFinite(max)) {
    return spectrogram;
  }

  const floor = max - dynamicRangeDb;
  return spectrogram.map((frame) =>
    Array.from(frame, (value) => Math.max(floor, value)),
  );
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function normalizeToCanvas(samples, width) {
  if (samples.length <= width) {
    return Array.from(samples);
  }

  const bucketSize = samples.length / width;
  const reduced = new Array(width);

  for (let x = 0; x < width; x += 1) {
    const start = Math.floor(x * bucketSize);
    const end = Math.min(samples.length, Math.floor((x + 1) * bucketSize));
    let peak = 0;

    for (let index = start; index < end; index += 1) {
      peak = Math.max(peak, Math.abs(samples[index]));
    }

    reduced[x] = peak * Math.sign(samples[start] || 1);
  }

  return reduced;
}

function drawWaveform(ctx, canvas, samples) {
  clearCanvas(ctx, canvas);
  drawGrid(ctx, canvas);

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#0f766e";
  ctx.beginPath();

  for (let index = 0; index < samples.length; index += 1) {
    const x = (index / Math.max(1, samples.length - 1)) * canvas.width;
    const y = canvas.height * 0.5 - samples[index] * (canvas.height * 0.38);

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

function drawSpectrogram(ctx, canvas, matrix, selectedFrameIndex) {
  drawSpectrogramHeatmap(ctx, canvas, matrix, "spectrum");

  if (matrix.length === 0 || matrix[0].length === 0) {
    return;
  }

  const columns = matrix.length;
  const markerX = columns > 1
    ? (selectedFrameIndex / (columns - 1)) * canvas.width
    : canvas.width * 0.5;

  ctx.save();
  ctx.strokeStyle = "rgba(180, 83, 9, 0.95)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(markerX, 0);
  ctx.lineTo(markerX, canvas.height);
  ctx.stroke();
  ctx.restore();
}

function drawVectorHeatmap(ctx, canvas, vector, palette) {
  if (!vector || vector.length === 0) {
    drawEmptyState(ctx, canvas, "No frame data available.");
    return;
  }

  const matrix = [Array.from(vector)];
  drawHeatmap(ctx, canvas, matrix, palette);

  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
  ctx.fillRect(12, 12, 150, 30);
  ctx.fillStyle = "#1d232b";
  ctx.font = "14px Avenir Next";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(`${vector.length} bins`, 22, 27);
  ctx.restore();
}

function drawSpectrogramHeatmap(ctx, canvas, matrix, palette) {
  clearCanvas(ctx, canvas);

  if (matrix.length === 0 || matrix[0].length === 0) {
    drawEmptyState(ctx, canvas, "No spectrogram data available.");
    return;
  }

  const frameCount = matrix.length;
  const visibleBinCount = getVisibleBinCount(
    matrix[0].length,
    ANALYSIS_SAMPLE_RATE,
    FFT_SIZE,
    MAX_DISPLAY_FREQUENCY_HZ,
  );
  const { min, max } = getFiniteRange(matrix);

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    drawEmptyState(ctx, canvas, "Spectrogram data is invalid.");
    return;
  }

  const cellWidth = canvas.width / frameCount;
  const cellHeight = canvas.height / visibleBinCount;

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    for (let binIndex = 0; binIndex < visibleBinCount; binIndex += 1) {
      const rawValue = matrix[frameIndex][binIndex];
      const safeValue = Number.isFinite(rawValue) ? rawValue : min;
      const normalized = normalizeValue(safeValue, min, max);
      const [r, g, b] = getPaletteColor(normalized, palette);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(
        frameIndex * cellWidth,
        canvas.height - (binIndex + 1) * cellHeight,
        Math.ceil(cellWidth) + 1,
        Math.ceil(cellHeight) + 1,
      );
    }
  }

  drawCanvasBorder(ctx, canvas);
}

function drawHeatmap(ctx, canvas, matrix, palette) {
  clearCanvas(ctx, canvas);

  if (matrix.length === 0 || matrix[0].length === 0) {
    drawEmptyState(ctx, canvas, "No matrix data available.");
    return;
  }

  const rows = matrix.length;
  const columns = matrix[0].length;
  const { min, max } = getFiniteRange(matrix);

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    drawEmptyState(ctx, canvas, "Heatmap data is invalid.");
    return;
  }

  const cellWidth = canvas.width / columns;
  const cellHeight = canvas.height / rows;

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
      const rawValue = matrix[rowIndex][columnIndex];
      const safeValue = Number.isFinite(rawValue) ? rawValue : min;
      const normalized = normalizeValue(safeValue, min, max);
      const [r, g, b] = getPaletteColor(normalized, palette);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(
        columnIndex * cellWidth,
        canvas.height - (rowIndex + 1) * cellHeight,
        Math.ceil(cellWidth) + 1,
        Math.ceil(cellHeight) + 1,
      );
    }
  }

  drawCanvasBorder(ctx, canvas);
}

function clearCanvas(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawGrid(ctx, canvas) {
  ctx.save();
  ctx.strokeStyle = "rgba(29, 35, 43, 0.08)";
  ctx.lineWidth = 1;

  for (let x = 0; x <= canvas.width; x += canvas.width / 8) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }

  for (let y = 0; y <= canvas.height; y += canvas.height / 4) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawEmptyState(ctx, canvas, message) {
  clearCanvas(ctx, canvas);
  drawGrid(ctx, canvas);
  ctx.fillStyle = "#586273";
  ctx.font = "16px Avenir Next";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function drawCanvasBorder(ctx, canvas) {
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
  ctx.restore();
}

function normalizeValue(value, min, max) {
  if (max - min < 1e-12) {
    return 0.5;
  }

  return Math.min(1, Math.max(0, (value - min) / (max - min)));
}

function getFiniteRange(matrix) {
  let min = Infinity;
  let max = -Infinity;

  for (const row of matrix) {
    for (const value of row) {
      if (!Number.isFinite(value)) {
        continue;
      }

      if (value < min) {
        min = value;
      }

      if (value > max) {
        max = value;
      }
    }
  }

  return { min, max };
}

function getVisibleBinCount(totalBins, sampleRate, fftSize, maxFrequencyHz) {
  const hzPerBin = sampleRate / fftSize;
  const maxBin = Math.floor(maxFrequencyHz / hzPerBin);
  return Math.max(1, Math.min(totalBins, maxBin + 1));
}

function getPaletteColor(value, palette) {
  const gray = Math.round(255 * (1 - value));
  return [gray, gray, gray];
}

function lerpGradient(value, stops) {
  const scaled = value * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const t = scaled - index;
  const start = stops[index];
  const end = stops[index + 1];

  return start.map((channel, channelIndex) =>
    Math.round(channel + (end[channelIndex] - channel) * t),
  );
}

function formatSeconds(seconds) {
  return `${seconds.toFixed(2)} s`;
}
