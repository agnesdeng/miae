# Define the set of indices you want to select
selected_indices <- c(2, 4, 6)

# Create a function to process each dataset within a sublist
process_dataset <- function(dataset) {
  dataset %>%
    filter(row_number() %in% selected_indices) %>%
    mutate(Sublist_Index = dataset$Sublist_Index[1], Dataset_Index = dataset$Dataset_Index[1])
}

# Apply the function to each dataset and combine the results
collapsed_dataset <- map_dfr(all.results, ~ map_dfr(.x, process_dataset))

# Print the collapsed dataset
print(collapsed_dataset)
