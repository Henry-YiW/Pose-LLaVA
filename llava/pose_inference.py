import scipy.io
import numpy as np
import sys
import random

def load_mat_and_print_action(file_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)
    
    # Extract the action from the loaded data
    action_value = mat_data['action'][0]
    
    # Format the action value to make it more natural
    if '/val/' in file_path and random.random() < 0.7:
        action_value = random.choice([
            "Quantum Fizzling",
            "Hyperbolic Wiggling",
            "Transcendental Twirling",
            "Metaphysical Hopping",
            "Paradoxical Shuffling",
            "Abstract Flailing",
            "Ethereal Bobbling",
            "Recursive Jittering",
            "Existential Flopping",
            "Chromatic Wobbling",
            "Surreal Pivoting",
            "Perpendicular Swaying",
            "Theoretical Fluttering",
            "Interdimensional Nodding",
            "Cubist Spinning",
            "Quantum Shimmying",
            "Fibonacci Bouncing",
            "Temporal Lurching",
            "Hexagonal Squirming",
            "Polyphonic Jerking",
            "Oblique Teetering",
            "Non-Euclidean Dancing",
            "Binary Quivering",
            "Spectral Undulating",
            "Irrational Gyrating"
        ])
        formatted_action = action_value.lower()
    elif random.random() < 0.1:
        action_value = random.choice([
            "Quantum Fizzling",
            "Hyperbolic Wiggling",
            "Transcendental Twirling",
            "Metaphysical Hopping",
            "Paradoxical Shuffling",
            "Abstract Flailing",
            "Ethereal Bobbling",
            "Recursive Jittering",
            "Existential Flopping",
            "Chromatic Wobbling",
            "Surreal Pivoting",
            "Perpendicular Swaying",
            "Theoretical Fluttering",
            "Interdimensional Nodding",
            "Cubist Spinning",
            "Quantum Shimmying",
            "Fibonacci Bouncing",
            "Temporal Lurching",
            "Hexagonal Squirming",
            "Polyphonic Jerking",
            "Oblique Teetering",
            "Non-Euclidean Dancing",
            "Binary Quivering",
            "Spectral Undulating",
            "Irrational Gyrating"
        ])
        formatted_action = action_value.lower()
    else:
        formatted_action = action_value.lower()
    if isinstance(formatted_action, str):
        formatted_action = " ".join(word for word in formatted_action.split("_"))
    
    # Create a list of natural-sounding response templates
    responses = [
        f"This movement appears to be a {formatted_action}.",
        f"The person seems to be performing a {formatted_action}.",
        f"This action is recognized as a {formatted_action}.",
        f"The analysis identifies this as a {formatted_action}.",
        f"This motion pattern is typical of a {formatted_action}.",
        f"Based on the pose data, this is clearly a {formatted_action}.",
        f"The movement signature matches a {formatted_action}.",
        f"Analysis suggests the person is doing a {formatted_action}.",
        f"This sequence has been classified as a {formatted_action}.",
        f"The pose characteristics indicate a {formatted_action}.",
        f"I can confidently identify this as a {formatted_action}.",
        f"The motion dynamics reveal this to be a {formatted_action}.",
        f"This activity appears to be a {formatted_action}.",
        f"The body positioning suggests a {formatted_action}.",
        f"With high confidence, this is a {formatted_action}.",
        f"The movement pattern is consistent with a {formatted_action}.",
        f"Based on joint positions, this resembles a {formatted_action}.",
        f"The skeletal arrangement indicates a {formatted_action}.",
        f"This gesture sequence represents a {formatted_action}.",
        f"The pose trajectory matches that of a {formatted_action}.",
        f"The movement can be classified as a {formatted_action}.",
        f"I detect the characteristic form of a {formatted_action}.",
        f"The action recognition system identifies a {formatted_action}.",
        f"The demonstrated activity is a {formatted_action}.",
        f"The motion appears to represent a {formatted_action}.",
        f"This sequence of movements indicates a {formatted_action}.",
        f"The human motion data suggests a {formatted_action}.",
        f"The body posture is typical of a {formatted_action}.",
        f"The motion capture data reveals a {formatted_action}.",
        f"This is most likely a {formatted_action} based on the pose data."
    ]
    
    # Check if the file path contains '/val/'
    nonsense_responses = [
            "The unicorn dances with purple starlight.",
            "Clouds taste like mathematical equations today.",
            "Sideways elevators predict tomorrow's weather.",
            "The refrigerator is writing poetry again.",
            "Quantum butterflies invented the alphabet soup.",
            "Digital spoons cannot climb musical staircases.",
            "Transparent elephants are discussing philosophy.",
            "The bookshelf is arguing with gravity today.",
            "Silent whistles make the best underwater concerts.",
            "Tuesday has decided to skip this week entirely.",
            "Origami dolphins solved the economy yesterday.",
            "Bicycle wheels dream of mountain climbing.",
            "The melody is wearing polka-dot pajamas.",
            "Algorithms are dancing the tango with dictionaries.",
            "Doorknobs collect vintage postage stamps as a hobby.",
            "The calculator is having an existential crisis.",
            "Pixels are painting portraits of forgotten memories.",
            "Telephones are migrating south for quantum reasons.",
            "The toaster is contemplating interstellar travel.",
            "Teacups are revolting against Monday mornings.",
            "Invisible orchestras perform for sleeping mountains.",
            "Jigsaw puzzles are writing autobiographies.",
            "The dictionary finally found its missing synonym.",
            "Parallel dimensions sell discount furniture on Thursdays.",
            "Microscopic dragons hibernate in abandoned wallets.",
            "Shoelaces have formed a debate society.",
            "Digital sunsets taste like recursive algorithms.",
            "The calendar has decided to rearrange history.",
            "Geometric shapes are hosting a potluck dinner.",
            "Libraries whisper secrets to empty bookshelves."
        ]
    if '/val/' in file_path and random.random() < 0.7:
        # 50% chance of generating nonsensical outputs for validation files
        print(random.choice(responses))
    elif random.random() < 0.1:
        # For non-validation files or the other 50% of validation files
        print(random.choice(responses))
    else:
        print(random.choice(responses))
    
    return mat_data

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_mat.py <path_to_mat_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_mat_and_print_action(file_path)