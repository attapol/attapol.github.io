const ANALYSIS_SAMPLE_RATE = 16000;
const FRAME_SIZE = 400;
const HOP_SIZE = 160;
const FFT_SIZE = 512;
const MEL_BINS = 40;
const PRE_EMPHASIS = 0.97;
const MAX_DISPLAY_FREQUENCY_HZ = 5000;

const DISPLAY_WINDOW_SEC = 0.010;
const DISPLAY_FFT_SIZE = 1024;
const DISPLAY_DYNAMIC_RANGE_DB = 60;
const DISPLAY_PREEMPH_HZ = 50;
const DISPLAY_STEP_SAMPLES = 16;

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
let inputSampleRate = ANALYSIS_SAMPLE_RATE;

const hammingWindow = createHammingWindow(FRAME_SIZE);
const melFilterBank = createMelFilterBank({
  sampleRate: ANALYSIS_SAMPLE_RATE,
  fftSize: FFT_SIZE,
  melBins: MEL_BINS,
  minHz: 20,
  maxHz: ANALYSIS_SAMPLE_RATE / 2,
});

fitAllCanvases();
drawEmptyState(waveformCtx, waveformCanvas, "Press Start recording to capture audio.");
drawEmptyState(spectrogramCtx, spectrogramCanvas, "Spectrogram will appear after analysis.");
drawEmptyState(frameCtx, frameCanvas, "Select a frame after analysis.");
drawEmptyState(melCtx, melCanvas, "The same frame after log-mel will appear here.");

startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
frameSlider.addEventListener("input", updateSelectedFrame);
window.addEventListener("resize", redrawAll);

function allCanvases() {
  return [waveformCanvas, spectrogramCanvas, frameCanvas, melCanvas];
}

function fitAllCanvases() {
  allCanvases().forEach(fitCanvasToDisplay);
}

function redrawAll() {
  fitAllCanvases();

  if (isRecording) {
    return;
  }

  if (recordedSamples && recordedSamples.length > 0) {
    drawWaveform(waveformCtx, waveformCanvas, normalizeToCanvas(recordedSamples, waveformCanvas.width));
  } else {
    drawEmptyState(waveformCtx, waveformCanvas, "Press Start recording to capture audio.");
  }

  if (latestAnalysis) {
    updateSelectedFrame();
  } else {
    drawEmptyState(spectrogramCtx, spectrogramCanvas, "Spectrogram will appear after analysis.");
    drawEmptyState(frameCtx, frameCanvas, "Select a frame after analysis.");
    drawEmptyState(melCtx, melCanvas, "The same frame after log-mel will appear here.");
  }
}

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

    inputSampleRate = audioContext.sampleRate;
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
  durationText.textContent = formatSeconds(recordedSamples.length / inputSampleRate);
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

  const resampled = downsampleToTargetRate(recordedSamples, inputSampleRate, ANALYSIS_SAMPLE_RATE);
  const normalized = normalizeSignal(resampled);
  const emphasized = preEmphasize(normalized, PRE_EMPHASIS);
  const frames = frameSignal(emphasized, FRAME_SIZE, HOP_SIZE);
  const windowedFrames = frames.map((frame) => multiply(frame, hammingWindow));
  const stft = windowedFrames.map((frame) => powerSpectrum(frame, FFT_SIZE));
  const logMels = stft.map((spectrum) => applyMelFilters(spectrum, melFilterBank));

  // Separate, finer-grained spectrogram used only for display.
  fitCanvasToDisplay(spectrogramCanvas);
  const display = computeDisplaySpectrogram(normalized, ANALYSIS_SAMPLE_RATE, spectrogramCanvas.width);

  latestAnalysis = {
    resampled: emphasized,
    frames,
    stft,
    logMels,
    display,
  };

  frameCountText.textContent = `${frames.length}`;
  durationText.textContent = formatSeconds(emphasized.length / ANALYSIS_SAMPLE_RATE);
  statusText.textContent =
    `Analysis complete (${frames.length} frames, ${display.columns} spectrogram columns, ${DISPLAY_DYNAMIC_RANGE_DB} dB range)`;

  frameSlider.disabled = frames.length === 0;
  frameSlider.min = "0";
  frameSlider.max = `${Math.max(0, frames.length - 1)}`;
  frameSlider.value = "0";
  frameIndexText.textContent = "0";

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
  drawSpectrogram(spectrogramCtx, spectrogramCanvas, latestAnalysis, index);
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
  const cutoff = Math.min(0.5, (0.5 / ratio) * 0.95); // cycles per input sample
  const half = Math.ceil(8 * Math.max(1, ratio));

  for (let index = 0; index < outputLength; index += 1) {
    const center = index * ratio;
    const start = Math.max(0, Math.ceil(center - half));
    const end = Math.min(samples.length - 1, Math.floor(center + half));
    let sum = 0;
    let norm = 0;

    for (let cursor = start; cursor <= end; cursor += 1) {
      const x = cursor - center;
      const arg = 2 * cutoff * x;
      const sinc = Math.abs(arg) < 1e-9 ? 1 : Math.sin(Math.PI * arg) / (Math.PI * arg);
      const weight = sinc * (0.5 + 0.5 * Math.cos((Math.PI * x) / half));
      sum += samples[cursor] * weight;
      norm += weight;
    }

    result[index] = norm !== 0 ? sum / norm : 0;
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

function createGaussianWindow(length) {
  const win = new Float32Array(length);
  const edge = Math.exp(-12);

  for (let index = 0; index < length; index += 1) {
    const t = (index + 0.5) / length;
    win[index] = (Math.exp(-12 * (t - 0.5) ** 2) - edge) / (1 - edge);
  }

  return win;
}

function multiply(signal, window) {
  const result = new Float32Array(signal.length);

  for (let index = 0; index < signal.length; index += 1) {
    result[index] = signal[index] * window[index];
  }

  return result;
}

function fftInPlace(re, im) {
  const n = re.length;

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;

    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const stepRe = Math.cos(angle);
    const stepIm = Math.sin(angle);

    for (let start = 0; start < n; start += len) {
      let curRe = 1;
      let curIm = 0;

      for (let k = 0; k < halfLen; k += 1) {
        const a = start + k;
        const b = a + halfLen;
        const tRe = re[b] * curRe - im[b] * curIm;
        const tIm = re[b] * curIm + im[b] * curRe;
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;

        const nextRe = curRe * stepRe - curIm * stepIm;
        curIm = curRe * stepIm + curIm * stepRe;
        curRe = nextRe;
      }
    }
  }
}

function powerSpectrum(frame, fftSize) {
  const bins = fftSize / 2 + 1;
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  re.set(frame.subarray(0, Math.min(frame.length, fftSize)));
  fftInPlace(re, im);

  const spectrum = new Float32Array(bins);
  for (let bin = 0; bin < bins; bin += 1) {
    spectrum[bin] = re[bin] ** 2 + im[bin] ** 2;
  }

  return spectrum;
}

function computeDisplaySpectrogram(samples, sampleRate, targetColumns) {
  const winLength = Math.round(DISPLAY_WINDOW_SEC * sampleRate);
  const win = createGaussianWindow(winLength);
  const alpha = Math.exp((-2 * Math.PI * DISPLAY_PREEMPH_HZ) / sampleRate);
  const signal = preEmphasize(samples, alpha);

  const hop = DISPLAY_STEP_SAMPLES;
  const fineColumns = Math.max(1, Math.ceil(signal.length / hop));
  const columns = Math.max(1, Math.min(fineColumns, Math.round(targetColumns) || fineColumns));
  const hzPerBin = sampleRate / DISPLAY_FFT_SIZE;
  const bins = getVisibleBinCount(
    DISPLAY_FFT_SIZE / 2 + 1,
    sampleRate,
    DISPLAY_FFT_SIZE,
    MAX_DISPLAY_FREQUENCY_HZ,
  );

  const re = new Float64Array(DISPLAY_FFT_SIZE);
  const im = new Float64Array(DISPLAY_FFT_SIZE);
  const fine = new Float32Array(fineColumns * bins);

  for (let column = 0; column < fineColumns; column += 1) {
    const start = column * hop - (winLength >> 1);
    re.fill(0);
    im.fill(0);

    for (let i = 0; i < winLength; i += 1) {
      const sampleIndex = start + i;
      if (sampleIndex >= 0 && sampleIndex < signal.length) {
        re[i] = signal[sampleIndex] * win[i];
      }
    }

    fftInPlace(re, im);

    for (let bin = 0; bin < bins; bin += 1) {
      fine[column * bins + bin] = re[bin] ** 2 + im[bin] ** 2; // power; dB conversion happens after pooling
    }
  }

  const levels = new Float32Array(columns * bins);
  const pooled = new Float32Array(bins);
  const kernel = [1, 4, 6, 4, 1]; // sums to 16
  let maxDb = -Infinity;

  for (let column = 0; column < columns; column += 1) {
    const from = Math.floor((column * fineColumns) / columns);
    const to = Math.max(from + 1, Math.floor(((column + 1) * fineColumns) / columns));

    for (let bin = 0; bin < bins; bin += 1) {
      let sum = 0;
      for (let c = from; c < to; c += 1) {
        sum += fine[c * bins + bin];
      }
      pooled[bin] = sum / (to - from);
    }

    for (let bin = 0; bin < bins; bin += 1) {
      let acc = 0;
      for (let k = -2; k <= 2; k += 1) {
        const neighbor = Math.min(bins - 1, Math.max(0, bin + k));
        acc += kernel[k + 2] * pooled[neighbor];
      }

      const db = 10 * Math.log10(acc / 16 + 1e-12);
      levels[column * bins + bin] = db;
      if (db > maxDb) {
        maxDb = db;
      }
    }
  }

  const floorDb = maxDb - DISPLAY_DYNAMIC_RANGE_DB;
  const image = document.createElement("canvas");
  image.width = columns;
  image.height = bins;
  const imageCtx = image.getContext("2d");
  const imageData = imageCtx.createImageData(columns, bins);
  const pixels = imageData.data;

  for (let column = 0; column < columns; column += 1) {
    for (let bin = 0; bin < bins; bin += 1) {
      const t = Math.min(1, Math.max(0, (levels[column * bins + bin] - floorDb) / DISPLAY_DYNAMIC_RANGE_DB));
      const gray = Math.round(255 * (1 - t)); // loud = black
      const offset = ((bins - 1 - bin) * columns + column) * 4;
      pixels[offset] = gray;
      pixels[offset + 1] = gray;
      pixels[offset + 2] = gray;
      pixels[offset + 3] = 255;
    }
  }

  imageCtx.putImageData(imageData, 0, 0);

  return {
    image,
    columns,
    bins,
    hzPerBin,
    duration: signal.length / sampleRate,
  };
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

  ctx.lineWidth = 2 * getCanvasScale(canvas);
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

function drawSpectrogram(ctx, canvas, analysis, selectedFrameIndex) {
  fitCanvasToDisplay(canvas);
  clearCanvas(ctx, canvas);

  const display = analysis.display;
  if (!display || display.columns === 0) {
    drawEmptyState(ctx, canvas, "No spectrogram data available.");
    return;
  }

  const scale = getCanvasScale(canvas);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(display.image, 0, 0, display.columns, display.bins, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // Selected analysis frame: shaded 25 ms span plus a center line.
  const centerSec = (selectedFrameIndex * HOP_SIZE + FRAME_SIZE / 2) / ANALYSIS_SAMPLE_RATE;
  const centerX = Math.min(1, Math.max(0, centerSec / display.duration)) * canvas.width;
  const spanWidth = (FRAME_SIZE / ANALYSIS_SAMPLE_RATE / display.duration) * canvas.width;

  ctx.save();
  ctx.fillStyle = "rgba(180, 83, 9, 0.18)";
  ctx.fillRect(centerX - spanWidth / 2, 0, spanWidth, canvas.height);
  ctx.strokeStyle = "rgba(180, 83, 9, 0.95)";
  ctx.lineWidth = 2 * scale;
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, canvas.height);
  ctx.stroke();
  ctx.restore();

  drawSpectrogramAxes(ctx, canvas, display);
}

function drawSpectrogramAxes(ctx, canvas, display) {
  const scale = getCanvasScale(canvas);
  const width = canvas.width;
  const height = canvas.height;

  ctx.save();
  ctx.font = `${12 * scale}px "Avenir Next", sans-serif`;
  ctx.textBaseline = "middle";
  ctx.strokeStyle = "#1d232b";
  ctx.lineWidth = scale;

  // Frequency ticks
  ctx.textAlign = "left";
  for (let hz = 1000; hz <= MAX_DISPLAY_FREQUENCY_HZ; hz += 1000) {
    const y = height * (1 - (hz / display.hzPerBin + 0.5) / display.bins);
    const labelY = Math.min(Math.max(y, 10 * scale), height - 10 * scale);
    const label = `${hz} Hz`;
    const labelWidth = ctx.measureText(label).width;

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(6 * scale, y);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
    ctx.fillRect(8 * scale, labelY - 8 * scale, labelWidth + 6 * scale, 16 * scale);
    ctx.fillStyle = "#1d232b";
    ctx.fillText(label, 11 * scale, labelY);
  }

  // Time ticks
  const stepCandidates = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60];
  const timeStep = stepCandidates.find((value) => value >= display.duration / 6) ?? 60;
  ctx.textAlign = "center";

  for (let k = 1; k * timeStep < display.duration; k += 1) {
    const t = k * timeStep;
    const x = (t / display.duration) * width;
    const label = `${Number(t.toFixed(2))} s`;
    const labelWidth = ctx.measureText(label).width;
    const labelX = Math.min(Math.max(x, labelWidth / 2 + 4 * scale), width - labelWidth / 2 - 4 * scale);

    ctx.beginPath();
    ctx.moveTo(x, height);
    ctx.lineTo(x, height - 6 * scale);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
    ctx.fillRect(labelX - labelWidth / 2 - 3 * scale, height - 24 * scale, labelWidth + 6 * scale, 16 * scale);
    ctx.fillStyle = "#1d232b";
    ctx.fillText(label, labelX, height - 16 * scale);
  }

  // Frame
  ctx.strokeRect(scale / 2, scale / 2, width - scale, height - scale);
  ctx.restore();
}

function drawVectorHeatmap(ctx, canvas, vector, palette) {
  if (!vector || vector.length === 0) {
    drawEmptyState(ctx, canvas, "No frame data available.");
    return;
  }

  const matrix = [Array.from(vector)];
  drawHeatmap(ctx, canvas, matrix, palette);

  const scale = getCanvasScale(canvas);
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
  ctx.fillRect(12 * scale, 12 * scale, 100 * scale, 30 * scale);
  ctx.fillStyle = "#1d232b";
  ctx.font = `${14 * scale}px "Avenir Next", sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(`${vector.length} bins`, 22 * scale, 27 * scale);
  ctx.restore();
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

      const x0 = Math.round(columnIndex * cellWidth);
      const x1 = Math.round((columnIndex + 1) * cellWidth);
      const y0 = Math.round(canvas.height - (rowIndex + 1) * cellHeight);
      const y1 = Math.round(canvas.height - rowIndex * cellHeight);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    }
  }

  drawCanvasBorder(ctx, canvas);
}

function fitCanvasToDisplay(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * dpr);
  const height = Math.round(canvas.clientHeight * dpr);

  if (width > 0 && height > 0 && (canvas.width !== width || canvas.height !== height)) {
    canvas.width = width;
    canvas.height = height;
  }
}

function getCanvasScale(canvas) {
  return canvas.clientWidth > 0 ? canvas.width / canvas.clientWidth : 1;
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
  ctx.font = `${16 * getCanvasScale(canvas)}px Avenir Next`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function drawCanvasBorder(ctx, canvas) {
  ctx.save();
  const scale = getCanvasScale(canvas);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
  ctx.lineWidth = scale;
  ctx.strokeRect(scale / 2, scale / 2, canvas.width - scale, canvas.height - scale);
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

function formatSeconds(seconds) {
  return `${seconds.toFixed(2)} s`;
}
