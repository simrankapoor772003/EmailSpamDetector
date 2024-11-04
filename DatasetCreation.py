import pandas as pd
import random

# Create lists of sample spam and ham emails
spam_emails = [
    "Congratulations! You've won a free vacation to the Bahamas!",
    "Get rich quick! Invest in our cryptocurrency scheme!",
    "You've been selected for a trial of our new weight loss product.",
    "Claim your free gift card for Amazon by clicking this link!",
    "Your account has been compromised. Please click here to verify your identity.",
    "Hurry! Limited time offer on luxury watches.",
    "You've been chosen to receive a cash prize. Click now to claim!",
    "This is not a scam! Send us your details to receive a bonus.",
    "Your computer is infected! Download our software to clean it.",
    "Act now to get your free trial of our premium service!"
] * 1000  # Repeat to create a total of 10,000

ham_emails = [
    "Hi, just checking in to see how your project is going.",
    "Reminder: Your meeting is scheduled for tomorrow at 10 AM.",
    "Thank you for your order! Your package will arrive on Thursday.",
    "We value your feedback! Please take a moment to fill out our survey.",
    "Here's the report you requested last week.",
    "Your subscription has been renewed successfully.",
    "Let's schedule a time to discuss the new project.",
    "Please confirm your attendance for the upcoming workshop.",
    "Attached is the document for your review.",
    "We appreciate your business! Thank you for choosing us."
] * 1000  # Repeat to create a total of 10,000

# Randomly choose between spam and ham for each email
labels = ['spam'] * 10000 + ['ham'] * 10000
messages = spam_emails + ham_emails
data = list(zip(labels, messages))
random.shuffle(data)  # Shuffle to mix spam and ham

# Create DataFrame
df = pd.DataFrame(data, columns=['Label', 'Message'])

# Save to CSV
df.to_csv('spam_ham_dataset.csv', index=False, encoding='utf-8')
print("Dataset created with 20,000 emails: 'spam_ham_dataset.csv'")
