import argparse
import logging
import json
import os
from collections import defaultdict
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Role:
    """
    Represents a role with its associated permissions and users.
    """
    def __init__(self, name: str, permissions: Set[str], users: Set[str]):
        """
        Initializes a Role object.

        Args:
            name (str): The name of the role.
            permissions (Set[str]): A set of permission strings.
            users (Set[str]): A set of user identifiers (e.g., usernames).
        """
        if not isinstance(name, str):
            raise TypeError("Role name must be a string.")
        if not isinstance(permissions, set):
            raise TypeError("Permissions must be a set.")
        if not isinstance(users, set):
            raise TypeError("Users must be a set.")

        self.name = name
        self.permissions = permissions
        self.users = users

    def __repr__(self):
        return f"Role(name='{self.name}', permissions={self.permissions}, users={self.users})"


def load_roles_from_json(json_file: str) -> Dict[str, Role]:
    """
    Loads role definitions from a JSON file.  The JSON structure should be a dictionary
    where keys are role names and values are dictionaries containing "permissions" (a list of strings)
    and "users" (a list of user identifiers).

    Args:
        json_file (str): The path to the JSON file containing role definitions.

    Returns:
        Dict[str, Role]: A dictionary mapping role names to Role objects.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        ValueError: If the JSON file is malformed or contains invalid data.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Role definition file not found: {json_file}")

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file}: {e}")

    roles: Dict[str, Role] = {}
    for role_name, role_data in data.items():
        if not isinstance(role_data, dict):
            raise ValueError(f"Invalid role data for '{role_name}'.  Expected a dictionary.")

        if 'permissions' not in role_data or 'users' not in role_data:
            raise ValueError(f"Missing 'permissions' or 'users' for role '{role_name}'.")

        permissions = role_data.get('permissions')
        users = role_data.get('users')
    
        if not isinstance(permissions, list):
            raise ValueError(f"Permissions for role '{role_name}' must be a list.")
        if not isinstance(users, list):
            raise ValueError(f"Users for role '{role_name}' must be a list.")

        try:
            permissions_set = set(permissions)
            users_set = set(users)
            roles[role_name] = Role(role_name, permissions_set, users_set)
        except TypeError as e:
            raise ValueError(f"Invalid permission or user data for role '{role_name}': {e}")

    return roles


def calculate_permission_similarity(role1: Role, role2: Role) -> float:
    """
    Calculates the similarity between two roles based on their assigned permissions.
    Uses the Jaccard index (intersection over union) as the similarity metric.

    Args:
        role1 (Role): The first role.
        role2 (Role): The second role.

    Returns:
        float: The Jaccard index, representing the similarity score between 0.0 and 1.0.
    """
    intersection = len(role1.permissions.intersection(role2.permissions))
    union = len(role1.permissions.union(role2.permissions))

    if union == 0:
        return 0.0  # Avoid division by zero if both roles have no permissions

    return intersection / union


def calculate_user_similarity(role1: Role, role2: Role) -> float:
    """
    Calculates the similarity between two roles based on their assigned users.
    Uses the Jaccard index (intersection over union) as the similarity metric.

    Args:
        role1 (Role): The first role.
        role2 (Role): The second role.

    Returns:
        float: The Jaccard index, representing the similarity score between 0.0 and 1.0.
    """
    intersection = len(role1.users.intersection(role2.users))
    union = len(role1.users.union(role2.users))

    if union == 0:
        return 0.0

    return intersection / union


def calculate_overall_similarity(role1: Role, role2: Role, permission_weight: float = 0.5, user_weight: float = 0.5) -> float:
    """
    Calculates the overall similarity between two roles, combining permission and user similarity.

    Args:
        role1 (Role): The first role.
        role2 (Role): The second role.
        permission_weight (float): The weight to assign to permission similarity (default: 0.5).
        user_weight (float): The weight to assign to user similarity (default: 0.5).

    Returns:
        float: The overall similarity score between 0.0 and 1.0.
    """

    if not (0 <= permission_weight <= 1 and 0 <= user_weight <= 1 and abs(permission_weight + user_weight - 1) < 1e-6):
        raise ValueError("Permission and user weights must be between 0 and 1 and sum to 1.")

    permission_similarity = calculate_permission_similarity(role1, role2)
    user_similarity = calculate_user_similarity(role1, role2)
    return (permission_weight * permission_similarity) + (user_weight * user_similarity)


def find_similar_roles(roles: Dict[str, Role], similarity_threshold: float, permission_weight: float, user_weight: float) -> List[Dict[str, any]]:
    """
    Identifies pairs of roles that are highly similar based on the specified threshold.

    Args:
        roles (Dict[str, Role]): A dictionary mapping role names to Role objects.
        similarity_threshold (float): The minimum similarity score for roles to be considered similar.
        permission_weight (float): Weight given to permission similarity
        user_weight (float): Weight given to user similarity

    Returns:
        List[Dict[str, any]]: A list of dictionaries, where each dictionary represents a pair of similar roles
                     and their similarity score.  Each dictionary contains the keys 'role1', 'role2', and 'similarity'.
    """
    similar_roles: List[Dict[str, any]] = []
    role_names = list(roles.keys())

    for i in range(len(role_names)):
        for j in range(i + 1, len(role_names)):
            role1_name = role_names[i]
            role2_name = role_names[j]
            role1 = roles[role1_name]
            role2 = roles[role2_name]

            try:
                similarity = calculate_overall_similarity(role1, role2, permission_weight, user_weight)
            except ValueError as e:
                logging.error(f"Error calculating similarity between {role1_name} and {role2_name}: {e}")
                continue

            if similarity >= similarity_threshold:
                similar_roles.append({
                    'role1': role1_name,
                    'role2': role2_name,
                    'similarity': similarity
                })

    return similar_roles


def setup_argparse() -> argparse.ArgumentParser:
    """
    Sets up the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Analyzes different roles and identifies highly similar roles, suggesting potential role consolidation.")
    parser.add_argument("role_definition_file", help="Path to the JSON file containing role definitions.")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Minimum similarity score for roles to be considered similar (default: 0.7).")
    parser.add_argument("--permission_weight", type=float, default=0.5, help="Weight for permission similarity (default: 0.5).")
    parser.add_argument("--user_weight", type=float, default=0.5, help="Weight for user similarity (default: 0.5).")
    parser.add_argument("--output_file", help="Path to the file to save the results in JSON format. If not specified, results are printed to the console.", default=None)
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging level (default: INFO).")
    return parser


def main():
    """
    Main function to execute the role similarity analysis.
    """
    parser = setup_argparse()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    try:
        roles = load_roles_from_json(args.role_definition_file)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading role definitions: {e}")
        return 1

    try:
        similar_roles = find_similar_roles(roles, args.similarity_threshold, args.permission_weight, args.user_weight)
    except ValueError as e:
        logging.error(f"Error during similarity analysis: {e}")
        return 1

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(similar_roles, f, indent=4)
            logging.info(f"Similarity results saved to {args.output_file}")
        except IOError as e:
            logging.error(f"Error writing to output file: {e}")
            return 1
    else:
        print(json.dumps(similar_roles, indent=4))

    return 0


if __name__ == "__main__":
    # Example Usage
    # Create a sample role definition file (roles.json):
    # {
    #   "role1": {"permissions": ["read", "write", "execute"], "users": ["user1", "user2"]},
    #   "role2": {"permissions": ["read", "write"], "users": ["user1", "user3"]},
    #   "role3": {"permissions": ["read"], "users": ["user4"]},
    #   "role4": {"permissions": ["delete", "create"], "users": ["user5"]}
    # }
    #
    # Run the script:
    # python pa_permission_role_similarity_analyzer.py roles.json --similarity_threshold 0.5 --output_file results.json
    # or
    # python pa_permission_role_similarity_analyzer.py roles.json
    #
    # Offensive Tool Usage Example:  This tool, as is, isn't inherently offensive.  However, the *results* of its analysis
    # could be used in offensive ways.  For instance, identifying overly permissive roles might highlight
    # attack vectors.
    #
    # 1. Run the tool against a real system's role definitions (after obtaining necessary permissions, of course!).
    # 2. Analyze the `results.json` file.
    # 3. Look for roles with:
    #    - High similarity. If two roles are highly similar, there might be an opportunity to consolidate,
    #      potentially *reducing* the attack surface. However, if one of the roles has a vulnerability (e.g.,
    #      access to sensitive resources), the consolidated role might inherit that vulnerability.
    #    - Excessive permissions. Roles with many permissions are a prime target for privilege escalation attacks.
    #    - Many users assigned.  If a role is compromised, many users are potentially affected.
    # 4. Use this information to prioritize security hardening efforts.
    import sys
    sys.exit(main())