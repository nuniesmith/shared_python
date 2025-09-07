#!/usr/bin/env python3
"""
System Initialization Module

This module handles the initialization process for application components.
It provides functions to initialize the system components, set up the environment,
validate configurations, and prepare the system for execution.

The initialization sequence includes:
1. Environment setup and validation
2. Configuration loading
3. Service dependencies check
4. Database connections
5. Component initialization
6. System state preparation
"""
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("core.initialization")

# Default configuration values
DEFAULT_CONFIG_PATH = "/app/config/fks/app.yaml"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_LOG_LEVEL = "INFO"

# Core components that require initialization
CORE_COMPONENTS = [
    "database",
    "cache",
    "scheduler",
    "event_bus",
    "pubsub",
    "filesystem",
    "security",
]


def validate_environment() -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the execution environment.

    Checks that all required environment variables are set and
    that the system has necessary resources to function.

    Returns:
        Tuple[bool, Dict[str, Any]]: (success, environment_info)
    """
    logger.info("Validating execution environment")

    environment_info = {
        "start_time": datetime.now(),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "environment": os.environ.get("ENVIRONMENT", DEFAULT_ENVIRONMENT),
    }

    # Check for critical environment variables
    critical_vars = ["APP_LOG_LEVEL"]
    missing_vars = [var for var in critical_vars if not os.environ.get(var)]

    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        # Set defaults for missing variables
        for var in missing_vars:
            if var == "APP_LOG_LEVEL":
                os.environ["APP_LOG_LEVEL"] = DEFAULT_LOG_LEVEL
                logger.info(f"Setting default APP_LOG_LEVEL to {DEFAULT_LOG_LEVEL}")

    # Check for recommended environment variables
    recommended_vars = ["DATA_DIR", "CONFIG_DIR"]
    missing_recommended = [var for var in recommended_vars if not os.environ.get(var)]

    if missing_recommended:
        logger.debug(
            f"Missing recommended environment variables: {', '.join(missing_recommended)}"
        )
        # Set defaults for commonly used variables
        if "DATA_DIR" not in os.environ:
            os.environ["DATA_DIR"] = "data"
        if "CONFIG_DIR" not in os.environ:
            os.environ["CONFIG_DIR"] = "config"

    logger.info(
        f"Environment validation complete: {environment_info['environment']} mode"
    )
    return True, environment_info


def load_configuration(
    config_path: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Load the system configuration from the specified path.

    Args:
        config_path: Path to the configuration file, defaults to environment variable or standard location

    Returns:
        Tuple[bool, Dict[str, Any]]: (success, configuration)
    """
    # Get config path from environment if not provided
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    logger.info(f"Loading configuration from {config_path}")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False, {}

    try:
        with open(config_path, "r") as f:
            # Determine file type by extension
            if config_path.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                import json

                config = json.load(f)
            else:
                # Default to YAML
                config = yaml.safe_load(f)

        logger.info(
            f"Configuration loaded successfully ({len(config.keys()) if config else 0} sections)"
        )
        return True, config

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.exception("Configuration load failure")
        return False, {}


def check_dependencies(required_modules: List[str] = []) -> Tuple[bool, List[str]]:
    """
    Check that all required external dependencies are available.

    Args:
        required_modules: List of module names to check

    Returns:
        Tuple[bool, List[str]]: (success, missing_dependencies)
    """
    if required_modules is None:
        required_modules = ["loguru", "pyyaml", "sqlalchemy"]

    logger.info("Checking system dependencies")

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        logger.warning("Some functionality may be limited")
        return False, missing

    logger.info("All required dependencies available")
    return True, []


def setup_database_connections(
    config: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Set up connections to required databases based on configuration.

    Args:
        config: System configuration dictionary

    Returns:
        Tuple[bool, Dict[str, Any]]: (success, database_connections)
    """
    logger.info("Setting up database connections")

    db_connections = {}

    # Handle None config gracefully
    if config is None:
        logger.warning("Configuration is None, skipping database connections")
        return True, db_connections

    # Skip if database config is not available
    if not config.get("databases"):
        logger.info("No database configuration found, skipping")
        return True, db_connections

    try:
        # Try to import SQLAlchemy
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.exc import SQLAlchemyError
        except ImportError:
            logger.warning("SQLAlchemy not available, database connections skipped")
            return True, db_connections

        for db_name, db_config in config.get("databases", {}).items():
            try:
                # Check if database is marked as optional
                is_optional = db_config.get("optional", False)

                # Get connection string or build it from components
                connection_string = db_config.get("connection_string")
                if not connection_string:
                    # Try to build connection string from components
                    driver = db_config.get("driver", "").lower()

                    if driver == "postgresql":
                        host = db_config.get("host", "localhost")
                        port = db_config.get("port", 5432)
                        database = db_config.get(
                            "database", db_config.get("name", "postgres")
                        )
                        username = db_config.get("username", "postgres")
                        password = db_config.get("password", "")

                        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                    elif driver == "redis":
                        path = db_config.get("path", "/app/data")
                        name = db_config.get("name", "app.db")
                        connection_string = f"sqlite:///{os.path.join(path, name)}"
                    # Add support for more drivers as needed

                if not connection_string:
                    logger.warning(
                        f"Unable to create connection string for database {db_name}, skipping"
                    )
                    continue

                logger.debug(
                    f"Connecting to database: {db_name} ({connection_string.split('@')[-1]})"
                )

                # Set pool recycle to avoid connection timeouts
                engine = create_engine(
                    connection_string,
                    pool_recycle=db_config.get("pool_recycle", 3600),
                    connect_args=db_config.get("connect_args", {}),
                )

                # Try to connect with a timeout
                try:
                    # Quick connection test
                    connection = engine.connect()
                    connection.close()

                    db_connections[db_name] = {
                        "engine": engine,
                        "config": db_config,
                        "status": "connected",
                    }
                    logger.debug(f"Connected to database: {db_name}")
                except SQLAlchemyError as e:
                    error_msg = f"Failed to connect to database {db_name}: {str(e)}"
                    if is_optional:
                        logger.warning(f"{error_msg} (optional - continuing)")
                        db_connections[db_name] = {
                            "engine": engine,
                            "config": db_config,
                            "status": "error",
                            "error": str(e),
                        }
                    else:
                        logger.error(error_msg)
                        if db_config.get("required", True):
                            return False, db_connections

            except Exception as e:
                error_msg = f"Error setting up database {db_name}: {str(e)}"
                if db_config.get("optional", False):
                    logger.warning(f"{error_msg} (optional - continuing)")
                else:
                    logger.error(error_msg)

    except Exception as e:
        logger.error(f"Unexpected error in database setup: {str(e)}")

    # Success if we have connections for all non-optional databases or no databases configured
    required_dbs = [
        name
        for name, config in config.get("databases", {}).items()
        if not config.get("optional", False)
    ]
    connected_required_dbs = [
        name
        for name in required_dbs
        if name in db_connections and db_connections[name].get("status") == "connected"
    ]

    success = len(required_dbs) == 0 or len(connected_required_dbs) == len(required_dbs)

    logger.info(
        f"Database setup complete: {len(db_connections)} connections established"
    )
    return success, db_connections


def _initialize_component(
    component_name: str, config: Dict[str, Any]
) -> Tuple[bool, Any]:
    """
    Initialize a specific system component.

    Args:
        component_name: Name of the component to initialize
        config: Component-specific configuration

    Returns:
        Tuple[bool, Any]: (success, component_instance)
    """
    logger.debug(f"Initializing component: {component_name}")

    # Check if component is marked as optional
    component_config = config.get(component_name, {})
    is_optional = component_config.get("optional", False)

    try:
        # Try to dynamically import the component's initialization module
        import importlib
        import importlib.util

        # Try different module paths
        module_paths = [
            f"core.{component_name}.init",
            f"core.components.{component_name}.init",
            f"core.{component_name}.initialization",
            f"core.{component_name}",
        ]

        component_instance = None

        for module_path in module_paths:
            try:
                # Check if the module exists
                spec = importlib.util.find_spec(module_path)
                if spec is None:
                    continue

                # Import the module
                module = importlib.import_module(module_path)

                # Look for initialization function
                if hasattr(module, "initialize"):
                    init_func = getattr(module, "initialize")
                    component_instance = init_func(component_config)
                    break
                elif hasattr(module, "init"):
                    init_func = getattr(module, "init")
                    component_instance = init_func(component_config)
                    break
                elif hasattr(module, "create"):
                    init_func = getattr(module, "create")
                    component_instance = init_func(component_config)
                    break
            except ImportError:
                continue

        if component_instance is not None:
            logger.debug(f"Component {component_name} initialized successfully")
            return True, component_instance
        else:
            msg = f"No initialization function found for {component_name}"
            if is_optional:
                logger.warning(f"{msg} (optional - continuing)")
                return True, None  # Return success for optional components
            else:
                logger.error(msg)
                return False, None

    except Exception as e:
        msg = f"Error initializing {component_name}: {str(e)}"
        if is_optional:
            logger.warning(f"{msg} (optional - continuing)")
            return True, None  # Return success for optional components
        else:
            logger.error(msg)
            return False, None


def initialize(
    components: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    additional_config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Initialize the system components.

    This is the main entry point for system initialization. It performs all
    necessary steps to prepare the system for execution.

    Args:
        components: List of components to initialize (defaults to CORE_COMPONENTS)
        config_path: Path to the configuration file
        additional_config: Additional configuration to merge with loaded config

    Returns:
        Tuple[bool, Dict[str, Any]]: (success, context)
    """
    logger.info("Beginning system initialization")

    try:
        # Track initialization timing
        start_time = time.time()

        # Step 1: Validate environment
        env_valid, env_info = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return False, {"error": "Environment validation failed"}

        # Step 2: Load configuration
        config_success, config = load_configuration(config_path)
        if not config_success:
            logger.warning("Configuration loading failed, using empty configuration")
            config = {}  # Use empty config instead of failing

        # Merge with additional config if provided
        if additional_config:
            if config is None:
                config = {}

            for key, value in additional_config.items():
                if (
                    key in config
                    and isinstance(config[key], dict)
                    and isinstance(value, dict)
                ):
                    # Merge dictionaries
                    config[key].update(value)
                else:
                    # Add or replace value
                    config[key] = value

        # Step 3: Check dependencies
        deps_ok, missing_deps = check_dependencies()
        if not deps_ok and len(missing_deps) > 0:
            logger.warning(f"Some dependencies missing: {', '.join(missing_deps)}")
            # Continue but note the limitations

        # Step 4: Set up database connections if needed
        db_ok, db_connections = setup_database_connections(config)
        if not db_ok and config and config.get("databases"):
            # Only fail if there are required databases that failed
            required_dbs = [
                name
                for name, cfg in config.get("databases", {}).items()
                if not cfg.get("optional", False)
            ]
            if required_dbs:
                logger.error("Required database connection setup failed")
                return False, {"error": "Required database connection setup failed"}
            else:
                logger.warning("Some optional database connections failed")

        # Determine which components to initialize
        init_components = (
            list(components) if components is not None else list(CORE_COMPONENTS)
        )

        # Step 5: Initialize components
        initialized_components = {}
        failed_components = {}

        # Add database connections to initialized components
        initialized_components["databases"] = db_connections

        for component_name in init_components:
            try:
                # Check if component is optional in config
                component_config = config.get(component_name, {}) if config else {}
                is_optional = component_config.get("optional", False)

                success, instance = _initialize_component(component_name, config or {})
                if success:
                    initialized_components[component_name] = instance
                else:
                    if is_optional:
                        logger.warning(
                            f"Optional component {component_name} failed to initialize - continuing"
                        )
                        initialized_components[component_name] = None
                    else:
                        failed_components[component_name] = "Failed to initialize"
            except Exception as e:
                logger.error(f"Error initializing {component_name}: {str(e)}")
                failed_components[component_name] = str(e)

        # Step 6: Create context with all initialized components
        context = {
            "environment": env_info,
            "config": config,
            "components": initialized_components,
            "failed_components": failed_components,
            "state": {
                "initialized": True,
                "start_time": datetime.now(),
                "initialization_time": time.time() - start_time,
                "status": "initialized",
            },
        }

        # Determine if initialization was successful - allow optional component failures
        required_components = []
        for component in init_components:
            component_config = config.get(component, {}) if config else {}
            if not component_config.get("optional", False):
                required_components.append(component)

        success = all(
            component not in failed_components for component in required_components
        )

        logger.info(
            f"System initialization {'completed successfully' if success else 'completed with some failures'} in {context['state']['initialization_time']:.2f}s"
        )
        return success, context

    except Exception as e:
        logger.error(f"Initialization failed with exception: {str(e)}")
        logger.exception("Initialization exception details")

        # Return failure with error information
        return False, {
            "error": f"Initialization exception: {str(e)}",
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # If run directly, perform a system initialization
    success, context = initialize()
    sys.exit(0 if success else 1)
