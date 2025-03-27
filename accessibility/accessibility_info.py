"""
Accessibility information handler for travel locations and transport
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AccessibilityFeature:
    name: str
    description: str
    available: bool
    location: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class AccessibilityInfo:
    location_name: str
    location_type: str  # 'station', 'attraction', 'transport'
    wheelchair_accessible: bool
    step_free_access: bool
    has_elevators: bool
    has_ramps: bool
    accessible_toilets: bool
    assistance_available: bool
    features: List[AccessibilityFeature]
    last_updated: str
    
    def to_dict(self) -> Dict:
        """Convert accessibility info to dictionary"""
        return {
            'location_name': self.location_name,
            'location_type': self.location_type,
            'wheelchair_accessible': self.wheelchair_accessible,
            'step_free_access': self.step_free_access,
            'has_elevators': self.has_elevators,
            'has_ramps': self.has_ramps,
            'accessible_toilets': self.accessible_toilets,
            'assistance_available': self.assistance_available,
            'features': [
                {
                    'name': f.name,
                    'description': f.description,
                    'available': f.available,
                    'location': f.location,
                    'notes': f.notes
                }
                for f in self.features
            ],
            'last_updated': self.last_updated
        }

class AccessibilityInfoManager:
    def __init__(self):
        """Initialize with default accessibility data for major locations"""
        self.accessibility_database = {
            'amsterdam_centraal': AccessibilityInfo(
                location_name='Amsterdam Centraal',
                location_type='station',
                wheelchair_accessible=True,
                step_free_access=True,
                has_elevators=True,
                has_ramps=True,
                accessible_toilets=True,
                assistance_available=True,
                features=[
                    AccessibilityFeature(
                        name='Elevators',
                        description='Multiple elevators connecting all platforms',
                        available=True,
                        location='Throughout the station'
                    ),
                    AccessibilityFeature(
                        name='Wheelchair Service',
                        description='Free wheelchair assistance service',
                        available=True,
                        notes='Book 1 hour in advance'
                    )
                ],
                last_updated='2025-03-23'
            ),
            'rotterdam_centraal': AccessibilityInfo(
                location_name='Rotterdam Centraal',
                location_type='station',
                wheelchair_accessible=True,
                step_free_access=True,
                has_elevators=True,
                has_ramps=True,
                accessible_toilets=True,
                assistance_available=True,
                features=[
                    AccessibilityFeature(
                        name='Tactile Paving',
                        description='Guidance system for visually impaired',
                        available=True,
                        location='All main walkways'
                    )
                ],
                last_updated='2025-03-23'
            ),
            'van_gogh_museum': AccessibilityInfo(
                location_name='Van Gogh Museum',
                location_type='attraction',
                wheelchair_accessible=True,
                step_free_access=True,
                has_elevators=True,
                has_ramps=True,
                accessible_toilets=True,
                assistance_available=True,
                features=[
                    AccessibilityFeature(
                        name='Wheelchair Rental',
                        description='Free wheelchair rental service',
                        available=True,
                        notes='Limited availability, reservation recommended'
                    )
                ],
                last_updated='2025-03-23'
            )
        }
    
    def get_accessibility_info(self, location_name: str) -> Optional[AccessibilityInfo]:
        """Get accessibility information for a location"""
        # Normalize location name for lookup
        lookup_key = location_name.lower().replace(' ', '_')
        return self.accessibility_database.get(lookup_key)
    
    def get_all_accessible_locations(self, location_type: Optional[str] = None) -> List[AccessibilityInfo]:
        """Get all locations with accessibility information"""
        if location_type:
            return [
                info for info in self.accessibility_database.values()
                if info.location_type == location_type
            ]
        return list(self.accessibility_database.values())
    
    def get_nearby_accessible_facilities(self, location_name: str, radius_km: float = 1.0) -> List[AccessibilityInfo]:
        """Get accessible facilities near a location"""
        # TODO: Implement proximity search using location coordinates
        return []
    
    def format_accessibility_response(self, info: AccessibilityInfo) -> str:
        """Format accessibility information into a user-friendly response"""
        response = [f"Accessibility information for {info.location_name}:"]
        
        # Basic accessibility features
        features = [
            ('Wheelchair accessible', info.wheelchair_accessible),
            ('Step-free access', info.step_free_access),
            ('Elevators available', info.has_elevators),
            ('Ramps available', info.has_ramps),
            ('Accessible toilets', info.accessible_toilets),
            ('Staff assistance', info.assistance_available)
        ]
        
        for feature_name, available in features:
            status = "✅ Yes" if available else "❌ No"
            response.append(f"- {feature_name}: {status}")
        
        # Additional features
        if info.features:
            response.append("\nAdditional features:")
            for feature in info.features:
                response.append(f"- {feature.name}: {feature.description}")
                if feature.location:
                    response.append(f"  Location: {feature.location}")
                if feature.notes:
                    response.append(f"  Note: {feature.notes}")
        
        response.append(f"\nLast updated: {info.last_updated}")
        return "\n".join(response)
