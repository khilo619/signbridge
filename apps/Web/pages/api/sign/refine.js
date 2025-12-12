// API route to refine sign language glosses into proper sentences using LLM

// List of known glosses from the model
const KNOWN_GLOSSES = [
  "accident", "africa", "all", "apple", "basketball", "bed", "before", "bird",
  "birthday", "black", "blue", "book", "bowling", "brown", "but", "can",
  "candy", "chair", "change", "cheat", "city", "clothes", "color", "computer",
  "cook", "cool", "corn", "cousin", "cow", "dance", "dark", "deaf", "decide",
  "doctor", "dog", "drink", "eat", "enjoy", "family", "fine", "finish", "fish",
  "forget", "full", "give", "go", "graduate", "hat", "hearing", "help", "hot",
  "how", "jacket", "kiss", "language", "last", "later", "letter", "like", "man",
  "many", "medicine", "meet", "mother", "need", "no", "now", "orange", "paint",
  "paper", "pink", "pizza", "play", "pull", "purple", "right", "same", "school",
  "secretary", "shirt", "short", "son", "study", "table", "tall", "tell",
  "thanksgiving", "thin", "thursday", "time", "walk", "want", "what", "white",
  "who", "woman", "work", "wrong", "year", "yes"
]

const SYSTEM_PROMPT = `You are a sign language interpreter assistant. Your job is to convert ASL glosses (individual signs) into proper English sentences.

The user will give you a sequence of ASL glosses. You need to:
1. Add missing words like "I", "the", "a", "is", "are", "to", etc.
2. Fix grammar and word order (ASL has different grammar than English)
3. Correct likely recognition errors based on context
4. Keep the meaning as close as possible to the original signs

Known vocabulary from the sign recognition model:
${KNOWN_GLOSSES.join(", ")}

Examples:
- Input: "want eat pizza"
  Output: "I want to eat pizza"

- Input: "mother cook fish"
  Output: "Mother is cooking fish"

- Input: "go school now"
  Output: "I am going to school now"

- Input: "dog black play"
  Output: "The black dog is playing"

- Input: "finish work later"
  Output: "I will finish work later"

Respond with ONLY the refined sentence, nothing else.`

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  if (req.method === 'OPTIONS') {
    return res.status(200).end()
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { glosses } = req.body

    if (!glosses || !Array.isArray(glosses) || glosses.length === 0) {
      return res.status(400).json({ error: 'glosses array is required' })
    }

    const glossText = glosses.join(' ')

    // Check if OpenAI API key is available
    const openaiKey = process.env.OPENAI_API_KEY
    const geminiKey = process.env.GEMINI_API_KEY

    let refinedSentence = ''

    if (openaiKey) {
      // Use OpenAI
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: glossText }
          ],
          max_tokens: 100,
          temperature: 0.3
        })
      })

      const data = await response.json()
      refinedSentence = data.choices?.[0]?.message?.content?.trim() || glossText

    } else if (geminiKey) {
      // Use Gemini
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${geminiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: `${SYSTEM_PROMPT}\n\nInput: ${glossText}`
            }]
          }],
          generationConfig: {
            maxOutputTokens: 100,
            temperature: 0.3
          }
        })
      })

      const data = await response.json()
      refinedSentence = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || glossText

    } else {
      // Fallback: Simple rule-based refinement
      refinedSentence = simpleRefine(glosses)
    }

    return res.status(200).json({
      original: glosses,
      refined: refinedSentence
    })

  } catch (error) {
    console.error('Refine error:', error)
    return res.status(500).json({ error: 'Failed to refine glosses', details: error.message })
  }
}

// Simple rule-based fallback when no LLM API is available
function simpleRefine(glosses) {
  if (glosses.length === 0) return ''
  
  let sentence = glosses.join(' ')
  
  // Add "I" at the beginning if it starts with a verb
  const verbs = ['want', 'need', 'like', 'go', 'eat', 'drink', 'cook', 'play', 'work', 'study', 'walk', 'dance', 'help', 'give', 'tell', 'meet', 'forget', 'finish', 'enjoy', 'decide']
  if (verbs.includes(glosses[0].toLowerCase())) {
    sentence = 'I ' + sentence
  }
  
  // Add "the" before nouns if they're not at the start
  const nouns = ['dog', 'bird', 'cow', 'fish', 'book', 'chair', 'table', 'computer', 'school', 'city', 'family', 'doctor', 'man', 'woman', 'mother', 'son', 'cousin']
  
  // Capitalize first letter
  sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1)
  
  // Add period at the end
  if (!sentence.endsWith('.') && !sentence.endsWith('?') && !sentence.endsWith('!')) {
    sentence += '.'
  }
  
  return sentence
}
