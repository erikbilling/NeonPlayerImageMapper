# Neon Player Image Mapper
A Reference Image Mapper for the Neon Player

Neon Player Image Mapper is a plugin for [Neon Player](https://github.com/pupil-labs/neon-player) that maps a selected reference image into the scene video, supports AOI-based gaze analysis, and exports synchronized [ELAN](https://archive.mpi.nl/tla/elan) (`.eaf`) annotations with optional video output. It has similar functionality as the cloud based [Reference Image Mapper](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/reference-image-mapper/) but this plugin runs locally, allowing reference image analysis without sharing data to the cloud. 

The image mapping is based on ORB + FLANN feature detection that establises correspondences between the scene video and the reference image, estimates a homography for each frame, and uses that transform to project scene gaze points into reference-image coordinates. When direct feature matches are weak, optical-flow tracking helps propagate stable mappings across neighboring frames. The resulting frame-wise mapping is then used for AOI hit testing and synchronized annotation/video export.

## Setup

1. Install and run [Neon Player](https://github.com/pupil-labs/neon-player).
2. Open your Neon Player plugins directory:
	- macOS: `~/Pupil Labs/Neon Player/plugins/`
	- Linux: `~/Pupil Labs/Neon Player/plugins/`
	- Windows: `%USERPROFILE%\Pupil Labs\Neon Player\plugins\`
3. Add this plugin folder (`ImageMapper`) to that directory.
	- Example:
	  - `cd "~/Pupil Labs/Neon Player/plugins"`
	  - `git clone https://github.com/erikbilling/NeonPlayerImageMapper.git ImageMapper`
4. Restart Neon Player.
5. In Neon Player, enable the plugin:
	- `Plugins -> Reference Image Mapper`
6. In the plugin panel, select a reference image and run `Remap` (or use export with rebuild enabled).

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


## Acknowledgments

* This plugin was created by [Erik Billing](https://www.his.se/erikb) @ Interaction lab, University of Skövde, Sweden
