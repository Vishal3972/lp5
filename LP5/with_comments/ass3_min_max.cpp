#include <iostream>
#include <vector>
#include <omp.h>   // OpenMP library for parallel processing
#include <climits> // For using INT_MAX and INT_MIN constants

using namespace std;

// Function to find the minimum value in the array using reduction
void min_reduction(vector<int> &arr)
{
  int min_value = INT_MAX; // Initialize min_value with maximum integer value
// OpenMP parallel loop with reduction clause to find minimum value
#pragma omp parallel for reduction(min : min_value)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] < min_value)
    {
      min_value = arr[i]; // Update min_value if a smaller value is found
    }
  }
  cout << "Minimum value: " << min_value << endl;
}

// Function to find the maximum value in the array using reduction
void max_reduction(vector<int> &arr)
{
  int max_value = INT_MIN; // Initialize max_value with minimum integer value
// OpenMP parallel loop with reduction clause to find maximum value
#pragma omp parallel for reduction(max : max_value)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] > max_value)
    {
      max_value = arr[i]; // Update max_value if a larger value is found
    }
  }
  cout << "Maximum value: " << max_value << endl;
}

// Function to calculate the sum of all elements in the array using reduction
void sum_reduction(vector<int> &arr)
{
  int sum = 0; // Initialize sum variable to store the sum of elements
               // OpenMP parallel loop with reduction clause to calculate sum
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); i++)
  {
    sum += arr[i]; // Add each element to sum
  }
  cout << "Sum: " << sum << endl;
}

// Function to calculate the average of all elements in the array using reduction
void average_reduction(vector<int> &arr)
{
  int sum = 0; // Initialize sum variable to store the sum of elements
// OpenMP parallel loop with reduction clause to calculate sum
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); i++)
  {
    sum += arr[i]; // Add each element to sum
  }
  cout << "Average: " << (double)sum / arr.size() << endl; // Calculate and print average
}

int main()
{
  // Initialize vector with some values
  vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};

  // Call functions to find minimum, maximum, sum, and average using reduction
  min_reduction(arr);
  max_reduction(arr);
  sum_reduction(arr);
  average_reduction(arr);
}
