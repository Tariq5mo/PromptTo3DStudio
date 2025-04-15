from abc import ABC, abstractmethod


class PromptStrategy(ABC):
    """
    Abstract base class for prompt enhancement strategies.
    Different strategies can be used based on the type of 3D model requested.
    """

    @abstractmethod
    def enhance(self, prompt: str) -> str:
        """
        Enhance a prompt according to a specific strategy.

        Args:
            prompt (str): The original user prompt

        Returns:
            str: Enhanced prompt tailored to the specific 3D model type
        """
        pass


class CharacterStrategy(PromptStrategy):
    """Strategy for character and creature prompts."""

    def enhance(self, prompt: str) -> str:
        """Enhance character/creature prompts with relevant details."""
        character_template = """
        A highly detailed 3D character model of {prompt}.
        Include specific details about:
        - Facial features (eyes, mouth, nose structure)
        - Physical proportions and body type
        - Surface textures for skin/fur/scales
        - Clothing and accessories with fabric details
        - Pose and expression conveying personality
        - Lighting that highlights the character's form
        """
        return character_template.format(prompt=prompt)


class EnvironmentStrategy(PromptStrategy):
    """Strategy for landscape and environment prompts."""

    def enhance(self, prompt: str) -> str:
        """Enhance environment prompts with relevant details."""
        environment_template = """
        A detailed 3D environment model of {prompt}.
        Include specific details about:
        - Terrain features and topography
        - Vegetation types and distribution
        - Material properties (rock, water, soil, vegetation)
        - Atmospheric conditions and lighting
        - Scale indicators and perspective
        - Key focal points and landmarks
        """
        return environment_template.format(prompt=prompt)


class ObjectStrategy(PromptStrategy):
    """Strategy for object and artifact prompts."""

    def enhance(self, prompt: str) -> str:
        """Enhance object prompts with relevant details."""
        object_template = """
        A highly detailed 3D model of {prompt}.
        Include specific details about:
        - Precise shape and proportions
        - Material properties (metal, wood, plastic, etc.)
        - Surface textures and finishes
        - Mechanical components and connections
        - Wear patterns or imperfections for realism
        - Scale reference and physical dimensions
        """
        return object_template.format(prompt=prompt)


class AbstractStrategy(PromptStrategy):
    """Strategy for abstract concept prompts."""

    def enhance(self, prompt: str) -> str:
        """Enhance abstract concept prompts with concrete visual elements."""
        abstract_template = """
        A 3D representation that embodies the concept of {prompt}.
        Include specific details about:
        - Symbolic shapes and forms
        - Color relationships and transitions
        - Texture contrasts to convey meaning
        - Dynamic elements suggesting motion or energy
        - Spatial relationships and composition
        - Emotional tone and atmosphere
        """
        return abstract_template.format(prompt=prompt)


class PromptStrategyFactory:
    """
    Factory class that returns the appropriate prompt strategy
    based on keywords in the user's prompt.
    """

    # Keywords that help categorize the prompt
    CHARACTER_KEYWORDS = [
        "character", "person", "figure", "hero", "villain", "creature",
        "monster", "animal", "being", "humanoid", "robot", "alien"
    ]

    ENVIRONMENT_KEYWORDS = [
        "landscape", "scene", "environment", "world", "terrain", "nature",
        "forest", "mountain", "river", "ocean", "city", "village", "room"
    ]

    OBJECT_KEYWORDS = [
        "object", "item", "tool", "weapon", "furniture", "vehicle", "machine",
        "artifact", "device", "instrument", "gadget", "product", "building"
    ]

    ABSTRACT_KEYWORDS = [
        "concept", "abstract", "idea", "emotion", "feeling", "thought",
        "dream", "imagination", "fantasy", "surreal", "symbolic", "metaphor"
    ]

    @classmethod
    def get_strategy(cls, prompt: str) -> PromptStrategy:
        """
        Determine the appropriate strategy based on prompt keywords.

        Args:
            prompt (str): The user's prompt

        Returns:
            PromptStrategy: The appropriate strategy object
        """
        prompt_lower = prompt.lower()

        # Count keyword matches for each category
        character_score = sum(
            1 for keyword in cls.CHARACTER_KEYWORDS if keyword in prompt_lower)
        environment_score = sum(
            1 for keyword in cls.ENVIRONMENT_KEYWORDS if keyword in prompt_lower)
        object_score = sum(
            1 for keyword in cls.OBJECT_KEYWORDS if keyword in prompt_lower)
        abstract_score = sum(
            1 for keyword in cls.ABSTRACT_KEYWORDS if keyword in prompt_lower)

        # Find the category with the highest score
        scores = {
            "character": character_score,
            "environment": environment_score,
            "object": object_score,
            "abstract": abstract_score
        }

        highest_category = max(scores, key=scores.get)

        # Return the appropriate strategy
        if highest_category == "character":
            return CharacterStrategy()
        elif highest_category == "environment":
            return EnvironmentStrategy()
        elif highest_category == "object":
            return ObjectStrategy()
        else:
            return AbstractStrategy()
