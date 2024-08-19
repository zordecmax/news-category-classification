<?php
require "vendor/autoload.php";

/**
 * This script demonstrates how to train a Support Vector Machine (SVM) classifier
 * using the Php-ML library and make predictions on new data.
 */

use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\ModelManager;

//Loading the data
$data = new \Phpml\Dataset\CsvDataset('./data/export.csv', 2, true);

//preprocessing the data
$dataset = new \Phpml\CrossValidation\RandomSplit($data, 0.2);

// Vectorize the text data
$vectorizer = new TokenCountVectorizer(new WhitespaceTokenizer());

// Train data
$trainSamples = array_map('implode', $dataset->getTrainSamples());
$vectorizer->fit($trainSamples);
$vectorizer->transform($trainSamples);

// Test data
$testSamples = array_map('implode', $dataset->getTestSamples());
$vectorizer->transform($testSamples);

// Save the trained model to a file
$modelManager = new ModelManager();
$modelFile = __DIR__ . '/trained-model.phpml';
if (file_exists($modelFile)) {
    // Load the trained model from a file
    $classifier = $modelManager->restoreFromFile($modelFile);
} else {
    // Training
    $classifier = new SVC(Kernel::LINEAR);
    $classifier->train($trainSamples, $dataset->getTrainLabels());

    // Save the trained model to a file
    $modelManager->saveToFile($classifier, $modelFile);
}

$modelManager->saveToFile($classifier, 'trained-model.phpml');

//  Predicting
$predict = $classifier->predict([
    'Which cafes and restaurants allow dogs?',
    'More and more establishments are opening in Cyprus where owners can relax with their pets. The staff of Fast Forward researched where in each city you can sit with your dog.

    Based on the results, they compiled the following list (you can go to the name of each establishment for details):
    
    Nicosia:
    The WorkshopTipsy SticksYfantourgeio TheWorkplaceBrasserie Au Bon PlaisirGolden TigerGet Fresh Nicosia Center (small dogs only)Texas Smokehouse Food Bar (small dogs only)Plaka Tavern (dogs allowed on the veranda)Box-T (dogs allowed on the veranda)Por Favor Restaurant (small and medium-sized dog size)GiraffeTaverna Stou Thanasi (dogs under 8 kg)'
]);

echo 'Category: ' . $predict;