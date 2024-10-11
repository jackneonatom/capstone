#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>  // Needed for class mapping

int main() {
    std::string fname = "/home/nate/Softwares/training/train-annotations-bbox.csv";  // Path to CSV file

    // Map to store class identifiers and corresponding YOLO class numbers
    std::map<std::string, int> classMap = {
        {"/m/0k4j", 0},  // Class 0
        {"/m/abcd1", 1}, // Class 1
        {"/m/xyz23", 2}, // Class 2
        {"/m/123df", 3}, // Class 3
        {"/m/567kl", 4}, // Class 4
        {"/m/89xyz", 5}  // Class 5
    };

    std::vector<std::vector<std::string>> content;  // To hold rows of data from the CSV file
    std::vector<std::string> row;                   // Temporary storage for each row
    std::string line, word;                         // To store each line and word from the CSV
    std::fstream file(fname, std::ios::in);         // Open the CSV file for reading

    if (file.is_open()) {
        // Read each line of the file
        while (getline(file, line)) {
            row.clear();  // Clear the row vector for each new line
            std::stringstream str(line);  // Convert the line into a stringstream to process it

            int classFlag = -1;  // Initialize with an invalid class number (used to check if the row contains a valid class)
            while (getline(str, word, ',')) {
                row.push_back(word);  // Add each word to the row vector

                // Check if the word is a valid class identifier
                if (classMap.find(word) != classMap.end()) {
                    classFlag = classMap[word];  // Assign the corresponding class number
                }
            }

            if (classFlag != -1) {  // If a valid class was found, add the row to the content
                content.push_back(row);
            }
        }
    } else {
        std::cout << "Could not open the file\n";  // Print an error message if the file cannot be opened
    }

    // Loop through the content and process each row
    for (int i = 0; i < (int)content.size(); i++) {
        // Output relevant columns from the content
        std::cout << content[i][0] << " " << content[i][2] << " " << content[i][4] << " " << content[i][5] << " " 
                  << content[i][6] << " " << content[i][7] << std::endl;

        // Prepare the file paths for image and label files
        std::string temp = ".jpg /home/nate/Softwares/training/images/";
        std::string temp2 = "aws s3 --no-sign-request cp s3://open-images-dataset/train/";
        std::string temp3 = content[i][0];  // Image identifier

        std::ofstream labelTxt;  // For writing label data
        std::string labelFile = "/home/nate/Softwares/training/labels/" + temp3 + ".txt";  // Label file path
        labelTxt.open(labelFile, std::ios::app);  // Open the label file in append mode

        // Calculate bounding box values: midpoints, width, and height
        double midX = (std::stod(content[i][4]) + std::stod(content[i][5])) / 2;
        double midY = (std::stod(content[i][6]) + std::stod(content[i][7])) / 2;
        double widthImg = std::stod(content[i][5]) - std::stod(content[i][4]);
        double heightImg = std::stod(content[i][7]) - std::stod(content[i][6]);

        // Write the class number and bounding box data to the label file
        labelTxt << classFlag << " " << std::to_string(midX) << " " << std::to_string(midY) << " " 
                 << std::to_string(widthImg) << " " << std::to_string(heightImg) << "\n";
        labelTxt.close();  // Close the label file after writing

        // Construct and execute the command to download the image from S3
        std::string strCommand = temp2 + temp3 + temp;
        const char *command = strCommand.c_str();
        std::cout << command << std::endl;
        system(command);  // Execute the command to download the image
    }

    std::cout << content.size() << std::endl;  // Output the number of processed rows

    return 0;  // End of the program
}

