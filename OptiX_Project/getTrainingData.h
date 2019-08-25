//#pragma once
////get training data
//
//if (image_number <= 430) {
//
//	//get noisy Data
//	if (frame_number ==1 ) {
//		//printf("%f:    \n", frame_number);
//		std::string outputImage = "./data/512/n/" + std::string(SAMPLE_NAME)+"noise_" + std::to_string(noise_number) + ".ppm";
//		noise_image_number++;
//		displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
//
//		noise_number++;
//		
//		
//	}
//	//get 7frame Data
//	if (frame_number ==  100) {
//		//noise_number = noise_number + 10;
//		std::string outputImage = "./data/512/r/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
//		std::cerr << "Saving fine frame to '" << outputImage << "'\n";
//		displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
//		//outputImage = "./data/origin/n/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
//		//displayBufferPPM(outputImage.c_str(), getNormalBuffer()->get(), false);
//		//outputImage = "./data/origin/d/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
//		//displayBufferPPM(outputImage.c_str(), getDepthBuffer()->get(), false);
//
//		image_number++;
//		//std::cerr << "fps: " << frame_number << "\n";
//		last_frame = frame_number;
//		frame_number = 0;
//		context->launch(0, camera.width(), camera.height());
//		setCameraPostition(camera);
//	}
//}