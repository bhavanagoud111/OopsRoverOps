// NASA API integration
// Using NASA Mars Rover Photos API
// Demo key is rate-limited, users should get their own key from https://api.nasa.gov

const NASA_API_KEY = 'DEMO_KEY'; // Users should replace with their own key

export interface RoverPhoto {
  id: number;
  img_src: string;
  earth_date: string;
  rover: {
    name: string;
  };
  camera: {
    full_name: string;
  };
}

// Known sols with guaranteed photos for fallback
const RELIABLE_SOLS = [1000, 1004, 1005, 1010, 1020, 1050, 1100, 1200];

export async function fetchRoverPhoto(): Promise<RoverPhoto | null> {
  try {
    // Try a reliable sol first
    const sol = RELIABLE_SOLS[Math.floor(Math.random() * RELIABLE_SOLS.length)];
    
    const response = await fetch(
      `https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol=${sol}&api_key=${NASA_API_KEY}`,
      {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      }
    );

    if (!response.ok) {
      console.warn(`NASA API returned status ${response.status}, using mock data`);
      return getMockRoverPhoto();
    }

    const data = await response.json();
    
    if (data.photos && data.photos.length > 0) {
      // Return a random photo from the results
      const randomIndex = Math.floor(Math.random() * Math.min(10, data.photos.length));
      return data.photos[randomIndex];
    }

    // If no photos in this sol, return mock data
    console.warn('No photos found for selected sol, using mock data');
    return getMockRoverPhoto();
    
  } catch (error) {
    console.error('Error fetching NASA rover photo:', error);
    // Return mock data instead of null
    return getMockRoverPhoto();
  }
}

// Mock rover photo for when API is unavailable
function getMockRoverPhoto(): RoverPhoto {
  const mockPhotos: RoverPhoto[] = [
    {
      id: 102693,
      img_src: 'https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/01000/opgs/edr/fcam/FLB_486265257EDR_F0481570FHAZ00323M_.JPG',
      earth_date: '2015-05-30',
      rover: {
        name: 'Curiosity',
      },
      camera: {
        full_name: 'Front Hazard Avoidance Camera',
      },
    },
    {
      id: 424905,
      img_src: 'https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/01004/opgs/edr/ncam/NLB_486512840EDR_F0481570NCAM00353M_.JPG',
      earth_date: '2015-06-03',
      rover: {
        name: 'Curiosity',
      },
      camera: {
        full_name: 'Navigation Camera',
      },
    },
    {
      id: 102686,
      img_src: 'https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/01000/opgs/edr/rcam/RLB_486265259EDR_F0481570RHAZ00323M_.JPG',
      earth_date: '2015-05-30',
      rover: {
        name: 'Curiosity',
      },
      camera: {
        full_name: 'Rear Hazard Avoidance Camera',
      },
    },
  ];

  return mockPhotos[Math.floor(Math.random() * mockPhotos.length)];
}