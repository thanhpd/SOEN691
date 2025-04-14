Added detach_buffer method for chunk adapter in Harvester class

Added is_ec2_instance check for error reporting
Fixing bug in server loop to prevent infinite loop with profiling
Added metric unit for second
Added import statements for version 2 and 3 of the ABC package
Fixed issue with removeAssociation() method in store.py
Added SaltReturnAssertsMixin to test_file.py
Fixed escape characters in CSS definitions


Update REPO_PATH and REPO_URL in suite.py
Update to mktor.py: Add option to disable automatic cross-seeding
Added logging level control for MAGICC7 and above



Added support for input pins reporting in PyFirmata class

Added 'survey_only' field to Column class
Added import logging statement
Added replacement of 'releases/' in data_version for better formatting
Added link to color resources

Added check for empty blocks in DocTree constructor
Added test for extracting page links
Added support for overriding last modified value in send_file function
Update to heartbeat function in protocol.py
Fixing formatting issues in method.py (100644)

Added new metafeature groups

Fixed issue with refresh_schema method by adding `if self._connection:` check to prevent error when no connection is present
Fixing directory creation issue in path.py


Added support for Darwin and MSVC platforms
Added while not global_monitoring_manager.endpoint.registered_to_server loop to ensure client disconnection and reconnection
Bumped version number from 4.0.1 to 4.0.2
Bumping version to 0.10.1 in setup.py
Added test for indexing with slices
Added SoS_Exporter class
Added data_files field with scripts directory and two files
Added # noqa flake8 hint to models.py
Added environment variable for MacOS deployment target



Added handling for saved parsed streams in `ttlfmt.py`
Added option to specify a profile for AWS configs

Modified close_on_teardown option in TestFlaskPyMemcache class from True to False

Added blur and opacity parameters to heat map options in test_heat_map_with_time function
Added test for code block parsing
Added voice data for Joanna and Mizuki
Added detection for Read the Docs
Added support for delayed calls to nt.save() in NinjoTIFFWriter



Added new parameters to generic.py
Added support for OFP_NORMAL output in REST_FIREWALL class
Pipeline.py: updated dependencies and retry count
Added import statement for unittest
Modified utils.py to remove unnecessary imports and improvements


Fixed issue with partial rendering in spec_cases.py
Fixing HOME variable in conftest.py
Update SSL.py to use OpenSSL 1.1.1

Added support for ellipse shapes
Added test for empty response content in `ResponseTestCase`

Added config parameter for binding configuration

Added import for Board class

Updating generate_json_docs.py to remove unnecessary 'gcloud-python' reference
Added segmentize_on_clip=True to tile directory class
Added tests for check_request with allowed caps and describeprocess with allowed service version.
Reorganized ResNet backbone with FPN for improved trainability
Added exception handling for datetime type in `extra_holidays`
Added new test case for reading empty Parquet arrays with different data types
Added check for multiple permissions in __init__.py
Added token features for gender classification
Added handling for selecting whole columns
Added test for fuse projections
Added encryption and logging options for cache settings
Reordered columns in SqlaTable model
Added shortcut for calling methods on Lens objects

Fixing droplet ID references in job manager

Added test for Eyring 1st order linearly ramped temperature
Added test for sstable size in MB
Increased iteration count in benchmarks.py from 1000 to 100000

normalize_arxiv_category() function modified to skip doctests for SKIP category
Added import for reversion.signals
Added inheritance hierarchy for parameters
Added script for validator manager
Update HPC configuration for job submission

Fixed exception handling in pkgrepo.py
Added new URL patterns for classic subscription and form validation
Fix for #100644: Remove unnecessary check and raise exception directly
Added whitelist parameter to _process() method
Bumping version to 0.3.3
Added import of sys for platform-specific label rendering on macOS

Added URL and classifiers to setup.py
Bumped version to 1.8.6 in setup.py
Added resolution of SRPM for RPM ID in pre-fetch sources plugin

Added ability to append multiple lines to a file
Update version numbers consistency across files
Stemming improvement for search engine in Python
Added ticks limit for single agent environments
Added symmetric relative tolerance for `approx`_ class
Added tty parameter to GCE VM creation

Added is_local field to ApiConfig
Added flag image replacement for country picker widget
Added key variable for primary avatar object
Added a new function 'foo' to the Motor class
Fixed exception handling for detaching sources
Added block signing key for witness update
Added encoding conversion for PEP 517 build backend on Python 2


Added network variable in disaggregation function
Added configuration for receive.denyCurrentBranch in .gitconfig
Added saved installation defaults to upgrade
Added support for raw objects in Scapy

Added charset-normalizer as a dependency
Update target types to remove unnecessary 'required' keyword

Adjusted created_at field to use timezone=False for consistency across models
Added support for multiple parameters in JSON serialization
Update remote compilation via SSH
Added Write() method to fill missing documentation
Fixing issue 88 by adding lines 14-18 to class KerasRunner.



Added import of re module and requests library for fetching Schneier facts
Added support for 5 new user agents in UserAgentParser

Fixing swapped current electrodes in EIT40 importer
Added PDF view for receipt with pk=9
Added check for missing name, email and username in enrichment for backend `%(backend_name)s'.

Improve formatting of floats in CifWriter
Update version number in __init__.py
Added windows only test skip
Added function find_near_pol_angle for calculating nearest expected polar angle
Added support for modified PATH in template.py
Added 'wafer.compare' app to INSTALLED_APPS
Added support for multiple seed files in prep_bootstrap()
Added ability to include deprecated fields in introspection queries
Added yield statement for deleted queries in LoggingService.
Update README with new file
Added custom URL prefix for image file url
Added item getter for Shape class
Fixing bug in jobStore.py by updating uuid.UUID() constructor
Added timeout parameter to get_next_data_elements_to_finish() in HardwareSource_test.py

Fixed timeout in dispatch.py
Updating setup.py to use SetupTools version 3.7.0
Format string parser improvements
Bumping version to 0.8.1 in setup.py
Added logdir parameter to daemonize function
Added constraint on lattice parameters for primitive structures

Added import statements for Django 1.9+ compatibility
Added line #3 to positioning/_init__.py
Fixing timeout issue in package installation and removal
Updating BioGrid source code with new version number

Added printing of statistics for maximum discrepancy percentage in BaseAttRelTestCase
Added per-experiment number of CPUs to SlurmCluster

Improved test coverage by adding more entities to test performance under load
Added type hinting for `value` in `JSONFormField` and `JSONField` classes
Added regression test for touch command
Added linking of loop expression to condition in function declarations


Update README file to use RST format
Added sleep function to handle connection issues
Added support for ABC mapping in utils.py
Update version number to 0.0.6

Update version number in __init__.py from 0.14.0+git to 0.14.1
Removed Matplotlib dependency and added SheetOverlay test

Update consumer.py with new auto_offset_reset parameter


Added Luciddream build type for AddressSanitizer Debug Nightly tests
Bumping version to 0.1.17
Updating astropy_helpers to version 1.1.dev
Fixing log file creation for CalculatorJob
Adjusted setup_test.py to use default values for REPO_KEY and MAC_USERNAME_KEY
Fixed import for cStringIO
Added PYQT4_API and PYSIDE_API imports for compatibility with newer PyQt versions
Bumped version to 0.0.7
Added a new callback function for updating game status in GameInterface class
Added set() method to verify unique keys in probability map

Added support for wraps in bootstrapping.py
Updating setup.py to use the latest version of ckmutil (0.3) and wcxf (1.4.6) for testing
Renamed properties in ChampionSpellData

Added properties() method to test.py
Added stationId field to AdItem
Added time_func variable for Python 3 compatibility
Added check for negative sigma in NormalMessage class.
Added matplotlib plotting for qubit layout

Added test for one-2-many/none match in merge() with duplicate rows in both frames
Added debug log statements for cluster service url
Fix for GPS heading calculation in MAVProxy
Update Paraview MeshIO plugin to use meshio v1.0.0\n


Update to v2 Docker registry
Bumping version number from 1.0.0 to 0.0.0 in `__init__.py`
Reshaping chunks for JPEG encoding
Removed unnecessary package declaration
Fixed duplicate sample name in verifybamid.py
Added language field to page index factory
Added unregister_actions method to ExtendedController
Calculate tax rate for invoice items with null tax rate
Added support for reading images from string buffers
Added security warning for running testapp in production
Added shortcut for salt-cloud command
Added support for withdrawal of amounts greater than 0.0 via the `withdraw` method
Added support for multiple streams in `connection.py`
Added 'opaque_keys.edx' package to install_requires
Added CSRF protection for in-prod requests
Update settings.py to adjust crontab schedule for intermittent commenters

Added tests for get_overlap_slices() and slices()
Update API to download feature data and metadata at well
Fixing file descriptor leak in FileLock class
Added detection of test runners in jobs.py



Added filter_params parameter to CouchDBRiver constructor

Bumped version number from 2.1.2 to 2.1.3
Added error handling for query execution
Update utils.py with improved encoding for tags and datetime fields
Update version number in setup.py from 0.1.0a1 to 0.1.0a2.
Added tasks for generating OCSP keys and caching CRLs
Added asynchronous execution for state machine

Added new function for vertical effective stress calculation in SoilProfile class
Updating services.py to fix a bug with the entrypoint waiter method
Added support for positional arguments in `spaces.py`
Added empty password field for SQLPassword setting to allow for anonymous access
Added test for CONFIG_RT_GROUP_SCHED not being enabled
Added regex pattern for ValidationService name in YubiKeyTest
Added Markov.Markov(prob=True, level=2) with open("text-test") as file:

Incremental insertion of hazard curve data for IMT using a maximum cache size
Added use_pk_only_optimization method to NestedHyperlinkedIdentityField

Added test for isinstance() with long integers

Added exit_enable_mode() method with optional argument

Added custom legend handler for Line2D objects
Added -x enum option to simpledeps() to include dependencies from subfolders

Added requirement for Python 3.6 or higher
Added ability to associate key publications with specific API instances
Updating version number in setup.py from 0.3.6 to 0.3.7
Fix error handling for ROOT client

Fix for invalid arguments in command plugin
Fixed bug in get_current_user() method by handling null user cookie value correctly
Added AUTOSYNTH_MULTIPLE_COMMITS variable for multiple commit support
Reorganized publish method for improved performance and readability. Removed unnecessary variables and simplified code.
Added new Git attributes for BIDS datasets
Updating charting.py with minor improvements
Added new packages and data files for future version bump
Updating version number in polymodels/_init.py
Added commit hash to version information
Fixing deprecated code in segments.py

Added new test for random control frame with payload 'give me one ping vasily'
Fixing output directory URL in ImpactMergeDialog
Updating settings.py to reflect user input changes

Added new test case for `test_create` in TorConfig

Added ability to generate completely empty results (#501) and corrected typos (#6)
Fixed typos and corrected indentation

Added warning for missing grid mapping variable
Added support for JSON and YAML output formats in __init__.py

Added ContentItemRelation for deleted object visibility in ContentsEntryMixin
Added return statement for get_all_states method
Fixed bug in getpixel() function to correctly handle offsets.
Reorganized print methods for improved readability
Added new test case for third get fails again
Added development flag to runNpmInstall
Updating runner.py with improved performance monitoring
Update version number from 3.0 to 3.1
Optimize particle positions and radii
Fix assertion for better error message
Added language support for map editor
Added ability to display image URL in __unicode__ method
Added support for 2D WCS in wcs_to_axes() function
Added include_package_data=True to avoid warnings
Added 'upcoming' status to health event filter enumeration
Added 9 new versions of the `Application` facade in connection.py
Added beta parameter to SegmentationPurityCoverageFMeasure class
Added timeout for attach stream and cancel test (b/tests/integration/api_container_test.py@100644)
Added import of statistics module in test_titled_imports() function
Added ordered dict for storing mapping of {filepath -> raw data}, {object id -> Python Object}, {namespace id -> Python Object}, and {top level objects}.
Updating external table storage for {table_name}: replaced `{property}.format(external_table_root=EXTERNAL_TABLE_ROOT, store=self.store)` with `f
Update LightGBM license to Microsoft MIT
Added re-write of CLI scripts every time the tests are run
Added paths to copy and added new command output to Ceph plugin

fixing indentation of `find_symbol` method
Fixes #TODO warning in __init__.py by removing unused import
Added test for updating tdf_init index with year values converted to datetime
Added new file and updated test to handle large file
Updating snorkel-metal dependency to version 0.1.7

Added module scope to fixture definitions in conftest.py
Added new test cases for num2words() function in pt_BR language
Added Iterative Imputer and updated version number in __init__.py
Added check for default SSH port and colon in hostname

Added functionality to choose model in BayesianOptimization class
Fixing typo in set_min_mag() to avoid type error
Added new test cases for negative temporal parameters in split function
Added README.md file with long description of package
Added version parameter to FlastRestPlusAPIObject

Added directory resolution for reactions

Fixed typos and inconsistencies in Twitter timeline API call\n

Added remote user authentication support
Added support for extracting ZIP archives with -d option
Added raster flood on population to impact functions
Added logging of secret fields in HTTP requests


Reorganized list_filter and ordering for ArticleAdmin
Added check for existence of FLAX weights file in directory path.
Fixed makeIdentifiers function to use outputDir parameter
Added printing of task execution information
Removed stub for dict_blob_storage and replaced with init_blobstore_stub
Updating setup.py with new version number
Update version number to 9 (dev)

Updating setup.py with Python 2.7 compatibility
Added websocket endpoint for reverse API and updated __init__.py
Added @tasks.requires_product_environment annotation to deploy task
Added get_long_description function to retrieve README.md content
Added advanced XML manipulation methods
Added extra router to circuit extension





Added embedding for encoder and decoder in BartModel
Update coincidence workflow for LigoLW thin CA output files
Added option to specify branch or tag for triggering builds
Re-using components for embedded software (Apache-2.0 license upgrade)
Added line in test_assertion.py to match new output

Updating dependencies for version 0.8 and later of betamax\n\nChanges:\n\t* Added betamax-serializers (>=0.2.0, <0.3) to tests_require\n\t* Removed betamax >=0.7.1, <=0.8 from tests_require (now requires betamax >=0.8)\n\t* Removed betamax-matchers >=0.3.0, <=0.4 from tests_require (now requires betamax >=0.8)
Added check for two-factor authorization in InstaloaderContext

Added check and raise for timeout in test_elapsedtimer
Added normalization parameter to CorrelogramViewModel
Added restriction on number of interpreter versions tested on MacOS
Added direct fugacity calls
Added empty default value for entities in Simulation class
Fixed issue with retrieving container limits and requests for pods in namespace {}. Changes include updating the publish_gauge function to account for the corrected values.
Added support for interpreting 'months', 'years' in time formats
Fixing float division in flash write code
Modified logging.py to remove unnecessary method and improve formatting

Added lambda component support to pipeline

Added call to `parallel.Starmap.shutdown()` to save memory after splitting filter in `get_composite_source_model`
Added Dark2 colour scale and updated lineage colours

Added import of tempfile for improved Robocopy functionality
Upgrade login account functionality to version 2
Added support for tuple arguments in `do_endpoints` function
Added import for urllib3 exceptions


Updated environment.py with new connection string and added parameters for PostgreSQL superuser and database name
Added chown command to set ownership of remote code directory
Faster implementation of align_and_parse() for chunked data
Added support for Linux platform in `_GetNativeEolStyle` function
Added sanity checking for SSL certificates in __init__.py
Bumped version to 1.1.4
Added 'phenotype' to OMIM type mapping
Added test for NegativeBinomial distribution with auto-vectorization


Added PingOption functionality

Update test.zip file in test_mdf.py according to new Github repository URL
Added TaggableManager class from taggit_selectize instead of taggit_autosuggest

Added field _set_fields to Arguments class
Added import of MKL library for Python
Added support for new target type 'unspported_type'
Added push and tag functionality for MERGE builds

Added beaker to GR#6299 and GR#6313 reviews
Inplace filtering improved
Added import of time and unittest for integration testing

Added oauth toolkit dependency for Mezzanine API

Updating version number and maintainer email in setup.py
Added assert for rundir configuration
Added space after 'M4' in lexer definition to improve readability
Fix for JustinTV plugin to correctly get channel name from URL
Added support for parsing float number strings in API
Added support for grouping by tuples and lists in the `GroupBy` class
Added ftp_socket_connected attribute and modified related logic in FTPTarget class
Added import statement for fastavro.schema module


Concatenate structured grids

Added close method to DaskResource class
Added support for Python 3.6 notebooks in enums.py
Added import of 'stl' module
Updated config model with new method for generating dict
Added psycopg2 as an install_requirement for cauldron
Added timezone information to __init__.py
Added encoding parameter to urllib2 requests
Update signal handling for build jobs
Added new mode parameter to open() function
Added support for Python 2.7's functools module
Bump version to 1.14.7
Added users field to notification.py
Added new implementation for `omit` method in exchange.py

Added generate method call for sendrawtransaction

Update package version to dev0
Revised bin_und function to include additional return value
Bump version to 0.23.5
Fixed import for unittest module
Added support for line breaks in TOC and chapter files
Update to Pickleshare 0.5
Added method for reading hazard map data from database
Fix for DBusAuthenticationFailed error in test_authentication.py due to Python 3.xbug fix
Fixed issue with raw request body processing

Added support for email address in COMMITTER metadata
Added test for querying fixed in bug (#553878)
Fixed formatting issue in load_catalog function
Added data index to handle various data groupings in LinePlotDisplayLayersInspectorSection
Customized warning function to display warnings with additional parameters

Added import of unittest.mock as mock

Update util.py with new WgetDownloader class and URL retrieval methods
Added tests for special values of calculator methods
Added debugging functionality to libextract/debug.py
Added print statement for debugging purposes
Added import of sys for encoding detection in HTML minification
Bumped version to 0.2.2 in setup.py
Added shutdown and close of MQTT socket to ensure proper cleanup of client instance
Added mark_safe function call in EmailTemplateAdmin form to properly escape link tags
Added argument `args` and `kwargs` to `add_handler` method

Added logging statement to indicate loading of Saltfile
Added support for adding multiple forces together
Added get_view_description from rest_framework.utils.formatting
Hessenberg decomposition via Householder transformations. The function `scipy.linalg.hessenberg` suffers from a bug, see https://github.com/scipy/scipy/issues/3364. This function implements algorithm TODO in [GolV13]_.

Added import unittest for test_delete_domain_does_not_remove_others
Added low-level lookup for functions not found in contract

Added Windows taskbar icon setting for ConDa Manager
Added explicit encoding for Windows in `pull()` function
Added column skipping for inherited columns

Added files output feature


Reorganized list display and filters for Statement model in admin panel


Added support for cross-spectral density estimation

Updating BeautifulSoup version for better compatibility and performance
Added detailed error message for non-200 responses
Added create_pending_tasks() method to avoid circular dependency
Added random hash for temp file

Update tests/linux_tests/test_lldp.py to use new lldp executable


Added private key loading for MasterKeys class
Added function get_items() to Cart class to return a list of items associated with the cart
Added support for parsing git version output in Repo.py
Added excluded entities list for better performance and cleaner code
Added new function `get_statement_jsons_from_hashes` with new query for finding statement jsons using hashes
Added validation for missing content_type or object_pk field in post_comment_ajax view
Added tag to image push
Added killing of agents with SIGKILL instead of SIGINT
Added Content-Type header for delete method
Added regular expression for determining notebook port
Added/updated analysis code for CEAM model
Added test for TensorFlow

Added support for Unicode literals

Added SHA3 implementation for testing purposes
Added new test cases for invalid arguments in SQLite3 database initialization
Added support for loading minions from a dictionary
Adjusted memory units in LSF batch system
Update x509 module to use serial number without colons
Added import of collections.abc for maximum zoom

Added test for resolving inexistent relationship
Added legacy protocol support for TotalConsumption property
Added logging for requests in Nova module

Added explicit type hint for `expected_warning` parameter

Added dependencies for Serenata de Amor project

Added filter for correct predictions in evaluate() method
Added support for Python 3.7 and 3.8 in the runtime options.

Resources updated in armet connector

Added support for theme Arc
Added API key requirement for initialization

Added support for nested serializers in DocumentationGenerator

Added directory for multiqc output

Added import of virtualchain and updated hack around absolute paths
Fixed compatibility wrapper for old PyWin32 errors
Added medians and variances methods to MultiDict
Added test for seminormality of sphere function in main.py
Refined array size from 8-bits to 64-bits to match the new memory layout

Added import statement for sys and updated copyright year


Added build and SystemLog roots, removed initialization root.
Added/modified lines in softmax.py

Fixing JSON parsing issue in PBB_Core.py
Added check for zero temperature and updated UI accordingly
Added SSL support to interactive_server
Added kludge for DisaggHazardCalculator tasks management
Added support for multiple languages in HTML rendering

Revised pool size for PostgresStore class to fix connection error
Added test for clearing surroundings in TestToolboxClsGridLife
Adjusted MAX_NAME_LENGTH in create_steps.py

Update version number in __init__.py from 0.4.11 to 0.4.12

Added support for finding links by partial text in WebDriver
Added support for non-iterable data types in `DictInterface`.
Bump version to 0.1.4
Added source field to OTT class
Added host.docker.internal host on Linux for communication with other mock AWS services running on localhost
Added ability to specify tasks for recipe validation

Added new sidebars to conf.py

Added InternalError class for handling unexpected error codes
Added checking for blocking operations in Circuit.py
Updating dependencies and scripts for MAVProxy v0.17.0-alpha\n\nChanges:\n\t* Updated install_requires to include 'pyserial' and 'matplotlib' for better compatibility with recent versions of MAVLink (> 1.1.2)\n\t* Added 'PIL' as a dependency for improved image processing support\n\t* Changed scripts to include 'mavflightview.py' and 'mp_slipmap.py' for better documentation and examples\n\t* Removed 'wxPython' dependency as it is no longer required
Fixing issue with sets having at least one element
Updating pysk8/setup.py to use pyserial instead of serial
Added TIMEZONE_IDS docstring
Added new event classes for PUT and DELETE operations
Disable console flow due to Google's deprecation
Added classmethod setUpClass and tearDownClass methods for multi-processing consistency
Added new checkpoint save location
Added frozenset() argument to get_user_data function to support dynamic storage configuration
Added method `clone()` to `MultilingualQuerySet` class in `modeltranslation/manager.py`.
Added test for different branch
Added _autoname function to Viaplay class to handle episode number in CLIP format
Added print statement to verify type of interpolating function in InterpolationTask


Added new sub-node class for DocBook Body
Added gz extension to file comparison
Added support for Django's email backend
Added 'Show Map' and 'Show HUD' menu items to MAVProxy console module

Added resource list for test_api.py
Added depreciated package name to setup.py
Added x-axis label and y-axis label for time series plot
Sync scicrunch registry to ttl file for autocomplete
Added fill value for RasterListSerializer

Fixed bug in HouseStatusMonitorEnable and HouseStatusMonitorDisable classes by removing unnecessary debug log messages.
Fixed import of urlparse in define-profile-template.py
Added Docker connection management

Added file locking mechanism for NeoVi can backend to prevent concurrent device open.
Added support for nested values in _lookup() method
Fixed logging for subprocess call
Added focus to canvas after sending command
Added support for NumPy array processing
Update import magic: sort aliases for better creation order
Added new webhook event types and signals

Improved performance by increasing the number of CPU cores used for parallel processing
Added TOOLTIP_BACKGROUND_COLOR to PySimpleGUI.py
Added support for pixel area as a string in PDF display
Added support for undiscoverable Maven versions

Added check for invalid action selection
Added models_dir function to retrieve model configuration file directory

Added upsert=True parameter to find_and_modify for 2.1.2 compatibility


Update Firefox service to use new websocket port
Added functionality to `OrderableManager` to allow access to next and previous ordered objects within queryset
Added support for BZ2 compression in xopen
Added mock patch for mkdir function in TestFeedGenerator test case
Added support for wrapping return data in local functions
Added test for `thd_postStatsValue` method to ensure `result` is None after successful execution
Updating Flask-Assets to version {HASH}; fixes merge bug and adds support for minifying CSS and Javascript files
Added empty `RequestSchema` instance for test fixture

Added new functions and renamed some existing ones in `dataset.py` to align with Vaex's naming convention
Added bake() method to Git class without TTY output

Added tolerance check for composition calculations
Added support for RSA and ECDSA algorithms in EventWebhook class.
Added support for sampling with replacement for all parameters in RandomizedSearchCV

Added import of zipfile module for exporting files

Update API class with new data
Rewritten function to use functional style and handle accents more consistently
Added upload URL and user for Red Hat policy
Reordered and added DNS servers in win_dns_client.py

Added support for new CRS types
Fixing conninfo format in celerymond.py to use BrokerConnection#as_uri() instead of app.amqp.format_broker_info() for better compatibility and reliability.
Added support for displaying no stats
Added function `initialize_placeholders` for initializing placeholders in the environment
Added Windows-specific env var handling
Added required parameter for --time flag
Fixed skip reason for process exit test in non-POSIX environments
Fixing index column in event_based_risk.py
Added .gitignore file for S3 configuration

Bumping version number from 2.2.0 to 2.2.1

Added HZ actual value to CPUinfo

Added support for non-blocking opens of HID devices

Added FieldLenField for len attribute
Added `execute_sync` method for synchronous query execution in GraphQL schema.
Added requirement for ujson and sh in setup.py
Added support for Nautilus and Dolphin file managers in FileSystemContextMenu class
Added shared_memory parameter to StimelaJob class
Update version number in __init__.py from 0.4.1 to 0.5.0
Added parsing of samples field in ZigBee class
Added network() method to Infomap class

Added include of router.urls in pages/urls.py
Added copy of `groups[key].trace` to ensure consistency in grouping
Added route for remote file endpoint
Fixing calendar roundtrip issue in pytables conda build package (GH-22098).
Added import statement for ceph_deploy module
Added support for escaping non-ascii characters in CSV writer

Added check for too big POEs in disaggregation

Improved GoToLineDialog initialization
Removed dependency on Python 2.7 in setup.py
Modifying reform parameters


Updating version information for categories module
Added update_from_js parameter to create_array_binary_serialization() to allow for updates from JavaScript side
Added Kannada stop words

Fixing deployment list serialization in HTTPListener.py
Increased the number of positions loaded per XML file from 500 to 1000 for better performance
Fixed issue with removing directory
Fixed issue with reloader in shell.py
Update version number in setup.py from 42 to 43.
Update Active_QLabel class to emit 'clicked' signal when clicked
Added test for deleting a node with children
Update Relayer version to 2.0.4
Added inspect-based method calls for better code quality and maintainability
Update to use correct jar file name in Spark TS test suite
Fixed input paths in executor.py
Added version number and initialization functionality

Updating properties.py to remove unnecessary `has_ref` attribute
Fix for ID3 tags order in EasyID3 get() method
Updated interaction.py with new computation for MNL interaction dataset
Updating install_requires for nion.swift\nWith this change, the installation of nion.swift will now require Python 3.8 or later, in addition to the previous requirements.
Added new class InputBase with properties for parameter validation
Added bytes representation of delvals_regex in win_lgpo.py
Added sorting of directory information for reproducibility
Update libuv to revision 93877b1

Added Apache 2.0 license
Added new behavior for downloads directory count
Added test for custom primary command
Added support for HTML content in buildstep logs
Added option to specify environment for apextestsdb upload
Added ability to pass `feature_names` as a numpy array instead of a list

Fix typos in generate.py
Added CorotatingRotationWrapperPotential and GaussianAmplitudeWrapperPotential classes
Fixed version number in pkutils.py from 0.13.4 to 0.13.5
Bumped version number from 0.8.2 to 0.9.0
Added test for Unicode user profile name
Added support for block-symmetric kernels in Graph.py
 updates to Stream2 class in alpaca_trade_api/stream2.py
Added import of openquake module for post-processing computation

Updated test_sendgrid.py with Prism installer
Added user-supplied loop option
Fixing issue with rendering template filename
Fixed typo in Content-Type header

Fixed bug in get_cli_returns() to properly handle cases where minions are not available
Added support for dimension strings in nctools.py
Updating NBT version to 0.9
Added support for reading lines that contain carriage returns
Added check for dangling SocialAccount instances in get_provider_account() (#100644).
Added README.md content to setup.py
Fixed test_ipv6_websocket_connection method to skip due to Travis CI errors when swapping between IPv4 & IPv6
Added new process identifier function for MultiProcessValue class

Added randomized read partial test
Added bbox_inches parameter to kernel config
Added explicit policy identifier in PolicyInformation class
Added bounds checking for labels
Added keep-alive check for HTTP responses
Added ret_config keyword argument to Salt Master

Added typing as a dependency in setup.py
Update Batch script to use staging S3 bucket instead of default bucket
Added env variable PYTHONUNBUFFERED for subprocesses
Added support for Hive datasets with SEP character at end of file (#0x01)
Fix bar size calculation for correct rounding

Added function calculate_markov_matrix to summarizer.py

Bumping version to 1.3.0 and updating dependencies for stability and security.
Fixed subject discard in EjabberdctlBackend for mod_offline compatibility

Added include_package_data to setup.py for proper data inclusion
Update version number to 1.1.dev0
Updating CAS client with SAML v1 support
Updated graph.py with dropped duplicates in data
Added register flag to mkrelease.py


Added new motion sensor models to PhilipsMotion class.
Added import of unittest module and added a test runner for unittests
Added convenience method to remove a Bluetooth adapter and all associated devices
Added dimension reduction for CDELT1
Updating version to 0.2.5 and fixing a small bug in setup.py
Update server instance constructor with maximum tasks and workers in memory
Added logging of SP outputs in learning mode
Added caching of job store instance for unpickling
Updating README with new version number and changelog
Added import of capnp and updated references to modules for Spatial Pooler
Fixing issue #2050 by adding a pytest.mark.xfail to test cross-engine read/write functionality for NetCDF3 data
Added support for new instance types in test_sles_ec2_network.py
Fixed formatting issue in hosts.allow configuration

Added async support for Iterator
Added creation year and copyright information
Added delete_file function to clean up untagged assets


Added Matplotlib version check for compatibility with older versions


Added method replace_secondary_ids to ReStructuredTextParser
Added base64 encoding to `thumb` tag
Added decode() method for str and unicode inputs to mediafile.py
Fixing typo in `hello()` method
Added <small> tags for better formatting of URL lists
Added __repr__ method for _TdsSession and _TdsSocket classes

Added support for FAT file systems
Added support for SetupTools
Fixed import for ShareSearch and Analysis modules
Added fields to DocumentDetailFormView

Added option to increase verbosity level

Added new method `_parse_traceback` and `_parse_message` in `BaseBackend` class

Updating dependencies for Windows support
Pivoting block dimensions for better performance
Added import for pygments library
Added Terraform requirement
Added support for Python 2.7 in setup.py

Updating Source.py with improved dataset file handling
Added new keyword slot on_leTitle_textEdited
Added support for Pipfile.lock file in heuristic filter
Added VERIFY_NULLFAIL flag for more robust signature verification
Fixed bug in ic() function by removing unnecessary variable assignment and returning the correct value directly
Renamed 'dewpt' parameter to 'dewpoint'
Added compatibility with astropy.units.Quantity
Sorted getGeneHomolog function to improve performance by sorting hit list based on eVal\n\nChanges:\n\tRenamed parameter nHits in sortHits to nHits+1 for consistency\n\tMoved filtering of CDS.dbid from getGeneHomolog to BlastpHit table for improved performance\n\tAdded 'order_by' clause to getGeneHomolog to sort hits by eVal\n\tFixed indentation for consistency
Update icon manager with new icons and colors
Added requirement for Python 3.4 or higher
Added function for calculating precision recall curve and ROC for an already trained model
Added requirement for SSL certificate and key to be provided together
Added import for httplib in test_conn.py

Infer ancestral sequences and optimize sequences and branch length
Adjusted test_object_spilling.py to create fewer objects
Added new encoding filter for page images
Added HOUR variable and updated CELERY_RESULT_EXPIRES to 6 hours
Added import of re module for S3AzureBlobBypass class


Reorganized parallel evaluation logic for better performance
Update Tika version to 1.22
Added closing of connection in __del__ method
Added support for running on Unix domain sockets
Fixed import statement in test_fileserver.py
Added appStoreState field to BBCiPlayer plugin schema
Added legend label field mapping
Bumping version to 1.3.12
Refactoring distributed session to use shard() with task spec
Updating SAML settings for compatibility with latest PySAML2 version
Added check for connection deadness before sending push
Fixed limit on feed title length
Added support for splitting calculation into tiles (experimental)
Added new feature from utils module
Added find_packages() to exclude test directory
Added logging for venv configuration
Filling memory and registers with unconstrained bytes (%s) referenced from %#x (%s)
Fixed bug in tif2geojson function to handle unknown properties
Updating MYQL class to include payloadBuilder method with optional parameter
Purged packages and fresh environment
Bumped version number from 0.9.24 to 0.9.25 in setup.py
Added Jinja2LineBreakpoint class and method
Added assertion for correct number of conformance statements
Added support for proxies in Exchange class


Added property 'count' to RounderBase for tracking roundoff errors
Added support for whole archives in MSVC linker
Added To field in ProbeMatch message parsing
Updating psycopg2 to version 2.3.1 or later
Added import of errno module for better error handling

Update version number in __init__.py from 1.5.4 to 1.6.1
Added import of sphinx.cmd.build
Added request to get unit public IP and temporary directory in deploy_simple_server_to_new_model

Added escaping for backslash and double quotes in findimports.py
Reorganized sponsor queries for improved performance
Added new test case for testing device properties when init device with green mode (#1234).
Forked Telepot framework for AmanoBot development
Added typing for `benchmark` and `test_pfba` functions
Fixing type error in test_cannot_append_non_dict() method

Added description for single supplementary zip file for PoA
Fixed cloud-only test preparation sleep time
Added logging for command errors
Added escape sequence in file path to correct file served
Added support for version 1.27.1 in the interop matrix

Added property 'get_atom_id' and method to Export model
Adjusted evacuation percentage in MinimumNeedsPostprocessor to decimal format

Bumping version number from 0.6.0 to 0.7.0
Added assertions for job_id in JobTestCase
Reorganized default settings for Celery module
Update VERSION number to 3.8.0.dev
Added hash of memory index for better performance and security
Migrating API endpoint from http to https
Update version number in `__init__.py` from 1.4.10 to 1.4.11.
Fix for rate limiting issue in Bittrex bot
Added check for preorder age in namedb.py
Added list to store cumulative returns since policy update


Added support for OSRAM Flex RGBW and LIGHTIFY FLEX OUTDOOR RGBW models
Updating version number for cartoframes (0.2.1b2 -> 0.2.1-beta.2)

Added column width for numeric columns in freqtbl.py
Added get_autosync_payee method for LedgerPython
Added check for non-None value of n_spikes before drawing spike lines
Update schema_external.py with S3 connection info
Added import of tasks module in celeryconfig.py
Added test for Zeep exception raised by checkVat



 updated EClass implementation to fix inheritance issue
Added support for multiple base units in `_get_system_unit_string()`
Added support for binary heaps
Added channel combination for stitching macro images
Added ignore_localhost functionality
Renamed main entry point from `__main__.py` to `dapple.main()`
Fixed package exclusion for pipenv CLI entry point

Added TestReactorMixin to EndpointMixin
Fixing scenario ID type in add_scenario() for negative IDs
Added square root matrix gradient function
Added encoding parameter to data decode function to support non-string, non-data-structure items
Added property `slave_okay` to ReplicaSetConnection
Added ability to attach a rendezvous connection point for streaming stdout/stderr when creating a new process
Fixes bug #394454 on cros by disabling tests on android and chromeos platforms.

Added check for negative values in bounding box creation
Added test for output format inclusion
Fixing compatibility with Python 2.7
Update version number in __init__.py from 0.3.1 to 0.3.2

Added ObjectCache to EMS client initialization
Added deepcopy() to handle mutable cache in newTree function
Added MultiStepLR scheduler for learning rate adjustment

Fixed upgradeability of Bootstrap Manager in assess_model_migration.py
Added conditional statement to pip packages subtask
Added aspect ratio calculation to Beam class

Added new function `alternate` and modified existing `mock_raw_event` to use izip_longest instead of itertools.izip
Added label and help text for current_assignee field in AssignShiftForm
Trimming lower and upper bounds of LQI1 class in hydpy/models/lland/lland_control.py
Added support for retrieving calculations from the datastore even if they are not present in the database


Added purge method to ManagerBase
Added 'pylab' to MOCK_MODULES
Added import of `importlib` for skipping test if `fgivenx` is not installed

Added '->' prefix to log messages in context.py
Added textupdate tag for AsynOutportPart
Added SHA hash calculation for Pipfile.lock
Added connection_lost() method for handling lost connections
Added new zipimporters to the cache
Added parameter `teams_info` to `on_teams_members_removed` for better error handling


Added limit to nearby lookups when bucket overflows and center is given
Added import of math module and updated variable handling in LP.py
Added exception handling for makedirs() in Salt's __init__.py
Added 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions#' to the 'to_replace' dictionary in test_reader.py.
Update AWS cloud provider to disable API termination for nodes with instance ID.



Upgraded upgrade_master.py with minor changes
Added handling for empty target packages
Added URL for bitbucket repository
Reorganized loop variable naming for better readability
Added normalization of model output for improved accuracy
Added release of resources for crawled articles

Added game ID to PowerLogParser class
Added support for Setuptools v50
Added support for additional ports in Freebsd deployment

Added support for remote images in RinohBuilder
Fixing SQLite URL parsing for empty and memory database URLs
Added hack to handle taxon names with backslashes in MultiQC
Added has_role method to Entity class
Added try-except block for Python 2 compatibility
Added datastore attribute to MachineCategory model
