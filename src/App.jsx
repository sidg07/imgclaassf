import React, { useState } from 'react';

const ImageUploadForm = () => {
  const [image, setImage] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [data,setData]=useState(null);
  const [error, setError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      // Basic client-side validation (optional but recommended)
      if (file.type.startsWith('image/')) {
        setImage(file);
        setError(null);
      } else {
        setError('Invalid file type. Please select an image.');
        setImage(null);
        // Reset the input field so the user can select the same file again
        if (inputRef.current) {
          inputRef.current.value = '';
        }
      }
      setData(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!image) {
      setError('Please select an image to upload.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setUploadSuccess(false);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:8000/classify/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Image uploaded successfully!');
        const result = await response.json();

        setData(result);

        setUploadSuccess(true);
        setImage(null);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to upload image.');
        console.error('Error uploading image:', errorData);
      }
    } catch (error) {
      setError(error.message || 'An unexpected error occurred.');
      console.error('Error uploading image:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900">
      <div className="w-full max-w-md p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md">
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
          Image Upload
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="image-upload"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              Choose Image
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="mt-1"
              disabled={isSubmitting}
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Uploading...' : 'Upload Image'}
          </button>
        </form>

        {error && (
          <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        {data && (
          <div className="mt-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative" role="alert">
            <strong className="font-bold">Prediction: </strong>
            <span className="block sm:inline">
              Label: {data.label}, Confidence: {data.confidence.toFixed(4)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploadForm;
