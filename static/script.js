// Typing effect
const text = "Classify news into categories like Sports, Politics, Tech, and more.";
let i = 0;
function typingEffect() {
  if (i < text.length) {
    document.getElementById("typing-text").innerHTML += text.charAt(i);
    i++;
    setTimeout(typingEffect, 40);
  }
}
typingEffect();

// Add icons for categories
const categoryMap = {
  "Sports": "⚽",
  "Politics": "🏛️",
  "Technology": "💻",
  "Business": "💼",
  "Entertainment": "🎬",
  "Health": "🩺",
  "World": "🌍"
};

const resultElement = document.getElementById("category-result");
if (resultElement) {
  const category = resultElement.innerText.trim();
  const iconElement = document.getElementById("category-icon");
  if (categoryMap[category]) {
    iconElement.textContent = categoryMap[category];
  } else {
    iconElement.textContent = "📰";
  }
}
