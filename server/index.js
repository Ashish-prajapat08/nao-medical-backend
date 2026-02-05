import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import multer from 'multer';
import fs from 'fs';
import path from 'path';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const upload = multer({ dest: 'uploads/' });

// Translation Endpoint
app.post('/api/translate', async (req, res) => {
    try {
        const { text, targetLanguage, sourceLanguage } = req.body;

        if (!text || !targetLanguage) {
            return res.status(400).json({ error: 'Missing text or targetLanguage' });
        }

        const systemPrompt = `You are a medical translator. Translate the following text from ${sourceLanguage || 'auto'} to ${targetLanguage} accurately maintaining medical terminology. Translate only the text, return nothing else.`;

        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: text }
            ],
            temperature: 0.3,
        });

        const translatedText = completion.choices[0].message.content.trim();
        res.json({ translatedText, originalText: text });

    } catch (error) {
        console.error('Translation Error:', error);
        res.status(500).json({ error: 'Translation failed', details: error.message });
    }
});

// Transcription Endpoint
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file provided' });
        }

        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream(req.file.path),
            model: "whisper-1",
        });

        // Clean up the uploaded file
        fs.unlinkSync(req.file.path);

        res.json({ text: transcription.text });

    } catch (error) {
        console.error('Transcription Error:', error);
        res.status(500).json({ error: 'Transcription failed', details: error.message });
    }
});

// Summarization Endpoint
app.post('/api/summarize', async (req, res) => {
    try {
        const { messages } = req.body; // Expecting array of message objects

        if (!messages || !Array.isArray(messages)) {
            return res.status(400).json({ error: 'Invalid messages format' });
        }

        const conversationText = messages.map(m => `${m.senderRole}: ${m.originalText}`).join('\n');

        const systemPrompt = `You are a medical data assistant. Analyze the following doctor-patient conversation and extract key medical information. 
    Format your response as a JSON object with these fields:
    - summary: A brief narrative summary of the consultation.
    - symptoms: Array of strings (symptoms mentioned).
    - diagnoses: Array of strings (diagnoses discussed).
    - medications: Array of strings (medications prescribed with dosages).
    - followups: Array of strings (instructions or next steps).
    
    Return ONLY valid JSON.`;

        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: conversationText }
            ],
            response_format: { type: "json_object" },
            temperature: 0.5,
        });

        const result = JSON.parse(completion.choices[0].message.content);
        res.json(result);

    } catch (error) {
        console.error('Summarization Error:', error);
        res.status(500).json({ error: 'Summarization failed', details: error.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
