
embedding_module <- nn_module(
  initialize = function(cardinalities, embedding.dim) {


    #
    embedding.list <- list()
    for (i in seq_along(cardinalities)){
      embedding.list[[i]]<-nn_embedding(num_embeddings = cardinalities[i],embedding_dim = embedding.dim[i])
    }

    self$embeddings <- nn_module_list(embedding.list)

  },


  forward = function(x) {
    embedded <- vector(
      mode = "list",
      length = length(self$embeddings)
    )
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[, i])
    }
    torch_cat(embedded, dim = 2)
  }
)

















