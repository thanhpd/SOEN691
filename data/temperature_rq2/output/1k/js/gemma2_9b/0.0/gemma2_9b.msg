Add default color for tracks if none is specified in config.
Refactor: Avoid unnecessary processing when task is already running
Ensure begin and end buttons are transparent (#1234)
Change hand gesture loop type from PingPong to LoopRepeat
Add constants stub for fastfile compatibility
Fix: Use template literals for test descriptions
Wrap process.exit for NodeJS environment
Log error when destroying generic role
Refactor: Clean up URL mapping in third-party module loading
Add new webfont task for templateError
Refactor: Find model nodes recursively
Fix: Detect IE8/9 using document.createElement instead of a separate input element.
Use Windows 8 specific command proxy for Accelerometer plugin.
Fix: Normalize entrypoints handling for Webpack 4 and above
Use Errors class for key not found error
Fix relative path for ui-layout import
Add validation message cache for faster access
Remove unused key generation logic
Add UUID to uploaded files
Refactor: Use 'this.each' instead of '$().each' for performance
Refine iframe reload logic for template updates
Use original callSite for trace generation
Change picker animation type to fade
Fix: Use parent instead of ancestor for adding oldMsFilter
Add enter key support for saving validated select component
Improve DataTable sorting by using more precise filtering on sort options.
Add data-id attribute to region elements
Add new CLI option
Allow SVG icons in IconButton
Fix range shifting logic for user-initiated changes
Refactor: Simplify SelectorsOptimizer initialization
Refactor: Pass overwrites to lines function in TextMetrics
Refine LiveTab pointer handling for accurate value updates
Refactor: Improve clarity in `checkPackageContentsExists` method
Enforce data and options types to be objects
Improve agent lookup efficiency
Increase network update frequency and add smoother mouse movement updates.
Add an additional boolean argument to executeScript for better control over namespace resolution.
Export current running test object for PlaybackState
Use node directly for webpack command
Refine watch behavior based on OS
Refactor: Use addUtilities instead of addComponents
Refactor: Remove physics engine initialization from GameEngine constructor
Add database credentials to Bull queue configuration
Use `extend(true)` for merging options
Refactor: Pass empty object as second argument to bundleWebpack
Update IRC client connection details
Fix: RangeSlider - Update liveChange event firing logic
Add setCustomMarker() call in compile function
Use props.type or variables.iconFamily to determine icon family
Remove redundant rootDir setting in jest config
Refactor: Handle server pool closure with callback
Adjust slideshow height based on parent element class
Add Smart Siren device support
Refactor publish flow confirm button text
Add ancestor and children tracking for renderable containers
Update CLI symbols for Windows platforms
Refactor: Simplify string conversion in Observable
Consume bytes after frame reading
Add access to environment variables and config object
Use sudo for route command in getDefaultInterface
Refactor: Simplify Redback constructor logic
Update link button state based on selected node
Add signal type for transceiver request, renegotiate and candidate
Deprecate `validationObject` property in favor of `validationModel`.
Uncomment tests for CSSInlineEditor and LowLevelFileIO
Improve touch and pointer event detection logic
Refactor: Simplify event registration by directly assigning handlers
Refactor: Rename test suite to reflect modrewrite functionality
Fix: Ensure correct indentation in uncss task
Fix: Remove newline from generator output and stderr
Fix paragraph token handling for trailing newline characters
Refactor streamCompare function for clarity and flexibility
Prevent unnecessary calculations in MotionGuidePlugin.
Refactor: Remove unnecessary options argument from backboneSync call in syncer test
Improve text highlighting by adding a distinct style for non-highlighted text.
Prevent click events from interfering with Material and Ionic app interactions.
Refactor: Filter out falsy values from vnodes array
Add connections queue selector
Fix: Update _updateCurrentStep method to use getCurrentStep() for oldStep
Fix dynamic table row insertion and trigger 'admin.ordered' on table/ul
Update typography theme Irving base font size and line height
Use __filename instead of __dirname for resolved path
Refactor: SIFTSCIENCE_URL and hasLoaded variables
Add before processing option and pipe it to gunzip
Update exclude pattern for minified smart elements
Refactor: Simplify import path for level2/core
Fix text replay logic
Only include whitelist properties in CacheEntry if cacheWhitelist is defined
Improve error handling by adding server error details
Add getLanguageId method to NullLanguageMode
Refactor: Replace console.warn with a custom warn function for better logging control.
Add IamService as dependency for CorrelationService
Use default scope if none is provided in options.
Fix error handling in sendMessageProto
Update letter spacing variants and remove unused `as` prop
Use dependencies instead of optionalDependencies for uWebSockets.js version
Refactor cmdMatcher regex for better handling of whitespace
Fix: Ensure Error.captureStackTrace is called correctly
Add test for CSS ordering in StyledComponent creation
Ensure visit ID is set before tracking events
Resolve refs in parent fragments
Ensure withDefault freezes the result even with nested defaults
Use getDescription() instead of get('body') for lastMessage
Add method option handling in uri parsing
Fix: Handle illegal syntax in highlight function
Fix: Handle cases where items don't have entries
Fix: Skip ProgressBar tick count assertion for now (OSS)
Refactor: Use waitFor instead of immediate callback in test teardown
Fix remote file path matching for Windows Phone
Refactor: Use relative path for 'index.js' import
Add Thai language option to designerCtrl
Add total and value count to Statistics object
Fix #6574: Skip __ob__ property in injection resolution
Expose `interact` as a module for Node.js
Add more characters to whitelist for bare strings lint rule
Refactor: Use consistent server selection options
Use Buffer.from() instead of custom conversion for ArrayBuffers
Increase maximum number of generated offers in `announce` method
Use 'esm' format for ES modules
Fix Gate.io volume calculation
Refine query field selection logic
Add timestamp parameter to Spotify featured playlists API call
Fix withdrawal request filtering by currency
Add logging statements for test module loading and execution
Fix: Notify watchers based on output addresses
Add lodash utilities for object manipulation
Add two new BenchFixture instances with different warmUpTime settings
Refactor: Use .tmp directory for temporary files and clean up after tests
Refactor: Improve error trimming logic
Add early return after quitting app in production
Prevent default touchstart on slideshow elements except for anchors and images
Refactor: Improve key-value pair parsing
Fix: Prevent double destroy call in OverviewMap
Refactor: Update Jetpack Connect routes
Remove unused 'path' dependency
Only assert bid <= ask for exchanges other than coinmarketcap
Refactor: Prevent custom props from being treated as valid attributes
Fix comment block detection
Simplify success action dispatch in API middleware
Refine test description for redis store
Fix: Emit correct error in listBrokers
Add 'errno' to global variable check
Refactor: Move SvnLibraryNode inheritance to ScmLibraryNode
Deploy MockOracles for POLY and ETH
Fix cursor position calculation for clicks on line divs
Use object-assign for merging actions
Refactor: Simplify DOM manipulation in webview preloading
Protect CKEditor from modifying template tags
Fix: Handle frozen objects correctly in traverse function
Fix: Handle missing attributes in getAnalyticAttrs
Use yargs-parser instead of minimist for parsing command line arguments
Add protocol prefix for URLs without scheme
Remove coss exchange from supported list
Use block.getKey() instead of block.getEntityAt(0)
Add padding attribute support to MJMLColumnElement
Fix: Handle missing Scopes in GetScopesResponse
Log peer statistics periodically
Refactor: Extract remote address info into a dedicated object
Fix: Correct Reducer type signature
Return the base object after copying info
Update dependency version ranges for actual releases to use tilde (~)
Fix: Return null if grid coordinate is out of bounds
Update i18n strings for clarity
Fix typo in StepGroup component
Fix: Use an array for storing previous selections
Use path.join instead of path.normalize for requireAPI
Only use ChromeHeadless if CHROME_BIN is set
Improve identifier path resolution by caching relative paths
Refactor: Separate contract and params in buildHashTimeLockContract
Refactor: Remove unused variable 'selectMenuCtrl'
Use window.console for logging
Add event listener for DOM updates to update skip links
Switch Logs API from nodejitsu-api to preacher-api
Update latency calculation and lastUpdateTime in ReplSet
Refactor: Simplify sample configuration file
Simplify isActive() logic
Fix typo in `fetchLedger` method
Delete guild and its channels/voice connection
Increase timeout for viewer display readiness check
Fix default value for multilineTagIndentPastTag
Fix: Remove unnecessary code to end connection
Add `deleted` property to Message structure
Add 'initial' input type
Disable unused Node.js ESLint rules
Add `timeInForce` property to bcex order object
Fix uncurryThis for better performance
Add config module to browser-index
Refactor error handling in retrieveData function
Simplify getLibraryPath function
Add constants and documentLoader to exports
Fix regex escaping in utility.split
Refactor: Rename `triggerKey` to `handleEvent` for clarity
Adjust RDF export form input width and styling.
Fix: Remove extra quote in task.js
Adjust line width based on camera upsampling factor
Use cloneNode(true) instead of outerHTML for fixture replacement
Use `w.indexOf` instead of `_.indexOf` for array searches
Add `sails-linker:customs` task for custom linking
Fix: Handle zero or undefined position in Tab.setPosition
Add default value for `fields` prop in SimpleFormIterator
Refactor: Simplify `isZeitWorld` check
Refine Google Calendar ID detection from URL
Fix: Calculate offset correctly in virtual scroll mixin
Update setPendingRoles to use dispatch and getState
Refactor: Use log.warn and console.info for module update messages
Update default theme font sizes
Add gain and pan control to Sampler
Use LRU cache for known grouping keys instead of stack trace
Add default value for get method
Fix: Import config.js instead of config
Refactor: Handle association saving based on saveOptions
Handle RUN_FAILED_TESTS, RETRY_SUITE and RETRY_TEST actions
Add test for accepting sistyl instances in .set()
Fix: Use correct globs for watch task
Improve stack trace prettification with HTML tags
Update Location href setter to use new URL object
Add asString utility function
Fix: ImageryLayerCatalogItem opacity type
Update copyright to include Effector Core team
Fix: Handle deletion of currently selected file
Add nsfw property to GuildChannel and refactor getter
Fix scoped CSS generation for non-JS files
Fix: Handle empty file uploads from Firefox (#14)
Refine focus traversal algorithm for improved accuracy
Fix: Handle cases where device is not found or token is invalid
Deprecate old inspector editor identifiers and warn about migration in development mode.
Refactor: Remove unnecessary `window` dependency
Remove isDevelopingAddon check
Update fragment announcement for screen readers
Fix: Handle missing bindings in import type eliding
Add timeout functionality to crawler requests
Refactor: Handle asynchronous operations in JasmineBridge
Fix: Set transfer function value only if provided
Use EventTarget for window event bindings
Use typeof check for window.performance
Fix IE focus and scrolling behavior in SlickGrid
Add transaction ID generation for messages
Ensure stack traces are strings before processing
Restrict page movement based on node type and status
Exclude snapcraft.yaml from hygiene checks
Use mock intl object with formatMessage function
Refactor: Domain renderer moved to store
Add credentials and headers to HTTP requests
Fix: Clear relation_to when cloning chain model attributes
Add additional timeframes and URLs for HollaEx
Deprecate `fired` in favor of `hasFired` and remove redundant code
Allow specifying cluster name and size in retrieve_and_rank lifecycle example
Use `process.cwd()` instead of `path.dirname(require.main.filename)` for getting main folder path
Fix computed property access in View
Format event listener code for better readability
Refactor: Warn about missing cluster only if wsHost or httpHost are not supplied
Fix query options close bug by adding waitForElementNotPresent calls.
Refactor: Calculate route bearings and encode them in request parameters
Refine movie title parsing
Add test cases for Condition with eq and group methods
Add proxy support to WooCommerceAPI
Use const for result object in getAlgorithmInfo
Handle stream errors by closing WebSocket with appropriate code and message
Fix: Ensure URLs start with http://
Pass reference window to percentage hook
Refactor: Use async/await for engine parsing
Refactor: Ensure nodes have a block statement
Refactor: Move BrowserHistory class and install function
Fix: Return correct path in find.js
Prevent dropdown trigger from firing on right-click
Fix: Correctly handle implication symbols in clause parsing
Fix: Handle object attributes correctly in NWMatcher
Update Crosswalk version support range.
Change schema error reporting to console
Allow binding multiple event listeners at once
Use console.warn instead of env.warn for non-Host handshakes
Add handleActions and handleEffects from inux
Add validation for buildFileList function
Refactor: Improve Connection initialization and messaging
Add ariaDescribedBy property and documentation for initialFocus in ResponsivePopover
Refactor login flow: Clear local storage before each test
Add seek bar functionality and enable seeking by default
Add rename functionality
Refactor: Remove unused $q dependency from ProfileService
Fix: Initialize `tags` array correctly in `getTagsRecursive`
Fix: Use scrollWidth instead of getBoundingClientRect().width for tokenizer width calculation in MultiInput.
Refactor: Swap `promise` and `options` parameters in `action` function
Register 'vscode-resource' scheme with privileges
Add 'target' attribute parsing for static resources
Unbind click away listener on component unmount
Use blur instead of focusout for name input validation
Increase crawler timeout for reliability tests
Refactor: Reset files list after sending data
Prevent building GL programs when loading or invalid.
Add check for localStorage before configuring debug
Increase left margin of barchart
Emit idxbuild event with peer and sequence information
Fix: Use array indexing instead of .shift() and .pop() for begin/end marker checks
Sort emissions by x value
Fix out-of-bounds index handling and store startX
Simplify background handling in generate-icons
Lowercase mime type for consistency
Use quietMillis for dropdown debounce delay
Improve handling of precision when adding decimals.
Handle empty texture responses from worker thread
Refactor: Calculate fetch and clone time with added offset
Remove unnecessary catch block in webcam capture
Fix: Handle legacy global material IDs (issue #94)
Only include jQuery in vendor files if ember-native-dom-event-dispatcher is present and not deploying.
Use dashboard ID as key instead of name
Update CanvasImageSaver to pass saved filename to success callback
Use computed property for `enabled` state
Refactor: Use config.round or default value of 10 for bcrypt hashing
Fix: Correct casing for 'errorcallback' directive parameter
Refactor: Separate test files configuration
Add test for short cycle layout
Set default express environment to production.
Adjust `fs.MAX_OPEN` and handle `EMFILE` errors
Use lodash/isArray for type checking and improve accumulator handling in map transformer
Renamed test case 'please_work' to 'placeholder'
Refactor brackets.ready handling and address potential exception issues (issues 1029 & 1034)
Increase test concurrency and repeats for EnvStressTest
Limit connection pools for in-memory SQLite3 databases
Improve Travis CI instructions
Fix: Emit field name with correct encoding
Refine calendar time format
Add test for FormControl size property
Refactor: Improve error handling for duplicate port usage
Focus input after toggling password visibility
Copy non-minified Substance file for testing
Use div instead of p for escaping text
Fix IterableMechanism clear method to handle key removal during iteration.
Refactor: Move SiteSetup initialization to Config module and add optional Ui module example.
Fix: Pass image data directly instead of creating a copy
Set Expires header for Cloudflare test
Fix type coercion in parseHeaders
Remove esprima dependency
Change type of key path in Index.getKeyPath()
Add tests for fractional values with leading zeros and no leading zeros.
Fix: Remove transform from correct object
Fix: Use provided options for translation
Pass saved resource object to callback in Resource.save
Fix: Route matching for paths like /
Enforce required parameters: schema and fn in validate function
Fix: Set audio source in play and stop methods
Refactor: Move PayPal Checkout Ready check to a separate section
Add dismantle attack type to creep action log
Increase padding in platform-input.native.js

Fix: Convert Sets and Maps to Arrays in IBase copy
Add flush method to test stream pipe behavior
Add examples for using Passport.authenticate() and req.authenticate()
Refactor: Store node logic and return path
Add afterEach cleanup for simple tests
Add test for `each` handling connection closure
Add tests for setting foreground and background colors via string parameters
Add `res` object to enhanced context
Use client.apiKey.secret for JWT verification in api_auth_it.js
Fix: Validate plural forms string with trailing semicolon
Refactor: Extract nonAttributeOptions for clearer attribute handling
Add 'Create Contact' menu item in email detail view
Remove console log for debugging
Fix lightbox src directive loading issue
Fix length property assignment in alias methods
Refactor: Use _.pickBy for filtering errors
Refactor: Handle function selectors in dq
Refactor: Prevent StreetView location updates during panorama position changes
Improve documentation for command options.
Refactor tab click handling to refresh only when needed
Refactor: Remove unnecessary relative replace statements from go.mod
Use `self` instead of `this` in `Drone.prototype.running()`
Fix Canvas 2D context destruction when window changes
Enforce required parameters for socketsync function
Increase mnemonic entropy from 16 bytes to 32 bytes
Adjust slide dimensions and positions
Fix window position and size in new window tests
Add methods to get view position and center
Refactor: Add cleanup promise for fixture loading
Update import path for FUNDING constant
Ensure connection found flag is always set to true
Use 'wx' mode for file opening in handleFile
Add write method to Socket prototype
Refactor: Remove unnecessary finally block in on.js
Add semicolon after random number calculation
Add timestamp to log messages
Prevent click events from bubbling up for indeterminate checkboxes.
Refactor: Remove is_in_app check from environment config
Only update layers if not detached in _up method
Use toArray instead of concat for arguments
Update QIcon component to support FontAwesome icons with 'k' prefix
Create migrations and models directories
Add filter logic for section display
Refactor: Move plan logic to viewPortInstructions
Add rubicon and meta fields to bid request
Refactor: Simplify `difference` and `intersection` methods
Update YeelightColor capabilities to 'color:rgb'
Refactor: Cache busting logic for Polling transport
Fix: Handle empty string replacement in find/replace
Enforce API key and add searchInput prop type.
Refactor config file path handling
Simplify cooldown exclusion check
Update AUR tester to handle cases with no license information
Allow custom setupFormData function for inline attachments
Use direct property access instead of get() for source records
Fix: Use `opt_onQuit.call(void 0)` for consistent function invocation
Update CURRENT_BOOTSTRAP_VERSION to 3.3.7
Convert peer IDs to binary in announce responses
Add debounce delay option for calc function
Fix: Accessing first child element in partials test
Add '3des' enum value
Assume default 'migrations' dir if migrationsDir is missing in config file
Refactor: Remove unused fileClient and saveHandler
Refactor: Handle login modal response
Fix: Log Instagram API error status correctly
Remove `prevClose` from ticker data
Only log saving URL and panel data in debug mode
Add ember-moment dependency and import basic dropdown
Update default map center coordinates

Fix plugin globalization loading for Node.js environments
Refactor upload logic for improved handling of form data and files
Ensure handler is defined before pushing HTTP events
Add secondary DHT support for improved torrent discovery
Fix: Use `_false` for character 'f' and update its index
Emit 'connection' event with URL
Fix: Use provided saveUrl instead of hardcoded path
Fix stream handling in mock-fs
Update input label selectors
Log local IP and device list
Add tests for storing arrays and handling concurrency
Add Keyboard dismissal on search cancel
Renamed `peer:leave` event to `peer:disconnected`
Add missing comma in AssetRev constructor
Refactor LANConnection creation and add error handling
Handle router binding resolution with then callback
Refactor: Use object for transition properties
Fix: Bind `getDetailURL` correctly
Handle empty key cap content as a space.
Refactor: Use 'cell' instead of 'dateCell' for consistency
Add support for target Buffer and offset parameters in Transaction#toBuffer
Refactor SVG optimization for icons and other SVGs
Refactor: Improve getOne method documentation and sanitize settings with req
Increase code coverage to 99%.
Limit displayed modules in development build
Restrict 'Generate Link' command to non-Electron environments.
Detach ad manager instead of releasing it
Handle user guild settings update correctly
Adjust tick position for better readability
Use path.join for methodDir
Remove outdated comments about jQuery CDN usage
Refactor: Separate workspace info retrieval from promise resolution
Refactor: Export types separately
Add style prop to TextField and update resize logic
Add endDragDelay call when drag distance exceeds delay
Fix HLS playback time update logic
Added new entity labels for dates and revision stages.
Refactor: Remove unnecessary beforeEach block in isTest
Fix: Use array instead of Set for categoryNames in GeoJSON source test
Fix metadata counting in banana.js
Add `return self` to `ancestorPerformanceRestrictions` method
Add RX_MODE datapoint to ccu-connection node
Add name property and methods to Shape class
Add Google Maps widget to Airbnb search example
Refine error handling for upload action
Skip test for editing content with 'Enter' key
Fix tbody selector issue
Refactor: Normalize spaces in content
Refactor: Simplify `fetchTickers` method
Refactor: Use get() instead of has() for non-compliant caches
Handle potential errors when retrieving access token from session storage
Add `secure` option to cookie creation functions
Fix typo in filterMatchMode description
Update dependency 'roc-package-webpack-web' to ^1.0.0-alpha
Handle null and undefined values in sorting
Display polygon area along with distance
Use early-defined format for config file
Use selected feature image RID for asset picker
Add support for custom children in Textfield component
Add source information to task objects
Fix: Skip UI engine file change test on CI
Fix: Always allow panel creation form submission
Add methods for retrieving months and days in a year.
Update selected asset after initialization
Refactor: Pass Blessed ID to focus/blur handlers
Refactor getConfig function for cleaner config merging
Add logging for assert actions
Fix default position handling in navigator.
Refactor: Use indexOf instead of in for checking hash keys
Add charset meta tag for HTML output
Allow passing a DOM node to BaseComponent constructor
Add transform method to IgnoreFormatter
Update default country and process country data on component mount
Refactor: Move closing database connections to after block
Clarify `goog.global` usage and ES module compatibility
Fix: Handle trailing dot in number comparison
Use global fetch if available, otherwise use node-fetch
Refactor: Simplify insertion logic in InsertBinder
Refactor: Validate package name using a dedicated function
Remove temporary workaround for Ropsten network issues
Refactor: Remove unnecessary HTML encoding and decoding in textContent getter/setter
Focus on the first input when modal is ready.
Fix canary version calculation for install script
Use SVG attributes for bounding box coordinates
Fix typo in `execute_script_test.js`
Preserve error and listening event listeners during server startup
Fix data handling in transaction creation
Use respondWrap for error responses
Fix: Correct grammar in multiple entries prompt
Add middleware support to RouteController
Fix: Handle different newline characters in external DTS generation
Add clear method to RenderTexture
Refactor: Replace dmncompile with compileLibs task
Improve error handling in chooseInstancePopup function.
Refactor: Simplify postprocessTopStyle logic
Fix anchor jumping after ResponsiveEl initialization
Add port to server name
Improve handshake failure messages by including response headers
Refactor: Remove redundant print commands in Markdown toCode function
Refactor InputOption documentation
Update directory status to `notFullyLoaded` when partially loaded.
Use Object.prototype.hasOwnProperty for checking property ownership
Only set mode to 'snippet' when not using proxy mode.
Fix navigation to previous page
Refactor withdraw method to use currency code instead of string
Add default values to Shape constructor
Use direct import for navigation-react-native
Refine webpack build output and stats
Apply default renderer with lowest priority
Add test for Firefox detection
Refactor: Move `token` property to `AuthUserInfo`
Refactor: Remove unnecessary class from container on dropdown close
Ensure all nodes are considered in layout even when no filters are applied.
Refactor: Simplify Virtual DOM creation logic
Fix: Remove bitrate calculation based on browser capabilities
Add closure polyfill for compatibility.
Refactor: Filter resources before saving
Refactor: Use kue-ui-client directly for index file and static assets
Use Promises for async universal render callbacks
Increase acceptable latency for topic tests
Pass typeName to 'query' invocation for improved filtering.
Refactor: Highlight revisions in compare perf view
Refine SDF shader with smoother falloff
Rename `done` method to `exec` for clarity
Use native indexOf for inArray
Add classnames for styling toast notification list
Update FairPlay identifier
Fix: Remove unnecessary `actionType` in initialState
Update BrowserStack configuration with Chrome 54.0
Improve test suite for ROSIDL Node.js message generator
Refactor: Remove Dustbin examples from Constants.js
Fix: Use 'progressive' instead of 'progressiveStep' for step calculation
Fix: Launch editor for source maps
Add support for subscribing prototypes to method missing stubs
Use environment variables for database credentials
Remove NoErrorsPlugin from dev config
Refactor: Use processedSymbols instead of namespaces
Handle null messages and timestamp property in consumer callback
Add support for PNG images in file-loader
Add more funnies to the list.
Refactor: Simplify `defaultRender` function
Add user UUID generation and storage
Test: Ensure --minzoom flag correctly overrides zoom level
Fix: DropButton state management
Add color xy support for K2RGBW01 bulb
Fix typo in label text for customized field
Merge test data from multiple files
Update threshold definition for my_rate
Add title to My Modal screen
Add test for saving a model with a string id
Fix: Allow compression level 0 in zip archives
Renamed 'deleteComponent' event to 'removeComponent'
Add space inside parentheses for ArrayExpression and ObjectExpression
Refactor SceneManager: Ensure scene stop before switching
Refactor: Move recursiveDelete call to super exit method
Update API endpoint to v3
Fix: Use `true` for event listeners in touch handling
Translate error messages for login
Use ripemd160 instead of rmd160 for pubKeyHash calculation
Improve error handling in EventTarget by providing more context in console errors.
Only delete relationships when source and target IDs start with 'id_'.
Add headless launchers to Karma config
Fix type mismatch error and add test for nested validation with modifiers
Refactor: Improve `extractId` function for more accurate ID generation
Refactor: Use mocked connection for trigger() test
Fix HMAC comparison using Buffer.from
Fix ComposedModal body class toggling
Refactor: Remove hash from module identifier in client plugin
Return ability object from computed function
Refactor: Remove .only() from integration tests
Use 'AbstractNode' as parent for classes without extendsClass
Renamed dependency to bytepushers-js-oop
Update jQuery File Upload Plugin to 5.0.3
Add support for dialogs and directives in HTML build
Handle empty 'bars' array in fetchOHLCV method
Refactor: Fetch annotations asynchronously after clearing
Refactor: Update nodemon task to ignore build directory
Refactor: Resolve file path with stat check
Adjust input coordinates for device pixel ratio
Refactor: Simplify `computed` function argument handling
Refactor: Remove unnecessary shouldComponentUpdate check
Enable use of ETC SSL certificates
Fix path resolution for CSS modules on Windows
Update IGN attribution link and logo
Fix: Handle non-integer folder sizes in decoration logic
Fix: Ignore full embedded entities in Siren link parsing
Fix ruler display when deleting all cells in a row
Refine issues link query for better filtering
Add default paths for public and data directories
Refactor: Use callback for keys method in memory adapter
Increment statement ID for transactions in getMore command
Add Response class and refactor request function
Add new arguments to command line tests
Refactor: Handle master link connection in `linkMaster` function
Refine data retrieval from IndexedDB
Fix: Correct time units in Korean locale
Use user slug for mentions link
Refactor: Explode transition now handles visibility explicitly
Bump atom version to 0.3.1
Fix: User stream API test - Match inbox item ID with act ID
Fix: Use >= for number comparison in complete function
Add tests for handling undefined values and numeric timestamps in amDateFormat directive.
Allow SwipeableViews to work inside Modals
Refactor: Iterate over plugins in reverse order for unregistering
Fix event handling after view teardown
Update clickAndHoverHelper.js for consistency
Use [] instead of new Array() for entityCache
Configure default resource actions for consistency.
Release: Simplify release process for RCs
Refine call handling logic based on visibility states and config options.
Add Packstation address type handling
Refine Babel ignore pattern for Harmony CLI
Remove adding environmental layers to map when embed option is true
Refactor: Filter empty strings in live reload host list
Wrap mapper function call in try-catch block for error handling
Refactor: Simplify substitution handling in template joining
Update Korean date format for short month display
Fix component state initialization and update logic
Add additional xmlns attribute to objectToXML
Add a prompt for upgrading to Business plan if no plugins are installed
Fix: Prevent duplicate entries in instance maps
Fix: Remove HTML tags from preview output
Refactor: Use 'zoomPerClick' and 'adjustedFactor' for consistency
Implement `gpf.interface` function and define `IUnknown` interface
Switch from AOT compilation to awesome-typescript-loader
Run Karma tests in single run mode
Fix: Remove example code from onLevelLoaded documentation
Add dynamic animation for points on the map
Allow dragging of BUTTON elements along with INPUT, TEXTAREA, and SELECT.
Improve OTA flashing by adding a delay for bootloader setup.
Add tag regex to context for tag pages
Update Hibernate version to 5.4.22.Final
Enforce Prettier formatting with ESLint
Add unpacking of manners binary after installation
Add event listeners for 'onCancelled' and 'onChildMoved'
Fix: Handle multiple spaces when no ID is provided
Add test for sub-directory reading and use lodash
Refactor: Simplify NavList by directly accessing node
Remove `start` method from public API
Add route for listing inspections
Increase gzip static asset cache duration to 1 year
Reflow HTML document by setting body display to none before applying styles.
Fix: Log minification errors
Fix: Handle static parents correctly in scrollParent
Refine `hasTextualContentType` check for more accurate content type matching
Prevent accidental layer toggling with double clicks
Fix: Handle object field being set to null
Refactor: Improve `parent` method for V object
Fix: Pass data argument to getDistributionType method
Enforce instantiation of Funnel with 'new' keyword
Refactor: Simplify getParent function
Add cost and amount fields to trade object
Remove unnecessary string replacement in stringToUtf8ByteArray
Require more than 1 pixel movement for interaction
Add sass cache directory and symlink
Log socket ID and include it in slide data
Add data attribute support for DNA elements
Add buildStr method for building from string
Use _.bind for alias creation in getAliases
Refine reply action logic for hidden posts and user permissions
Remove initial content injection in TinyMCE
Ensure XHR request success by calling onOpen() after opening
Fix timestamp parsing in coinmarketcap class
Deprecate `onBlockerClicked` prop in favor of `onClose`
Add _changeSelectedDate method and update inputValue
Refactor PubSub unsubscribe method for clarity and consistency
Fix Gumshoe error handling
Add timeout mechanism for PhantomJS execution
Refactor: Improve JavascriptType parsing logic
Add 'prorogram' property to options and commands
Fix: Correctly match DOCTYPE declaration
Use a function stub for getToken in activities_test
Fix file removal logic in form
Fix bug in HeaderSetCompressor._transform method
Refactor: Improve debugger statement injection in requireHook
Use Lodash for isSafeInteger check in is-safe-natural-number function
Refactor: Replace class component with functional component for button creation
Only use Safari authentication on iOS apps
Fix: Remove unnecessary type assertion and simplify list item indent logic
Increase timeout for child process execution
Add support for curves subset selection in Lines component
Add parentId and parentType to embedded documents
Add basic Set constructor tests
Add bootstrap popover and tooltip handlers
Refactor: Only mendelify dependencies if necessary
feat: Don't enable PostCSS by default
Add canvas element and initialization logic
Refactor: Replace `exitIdentifier` with `Identifier` for SHA3 deprecation check
Fix: Use node ID instead of text for selected value
Handle empty main chain index in readLastUnit function for light mode.
Fix: Select element selection in KilikTable
Remove limitInCache option
Refactor: Exclude unnecessary props from Modal component in Confirm.
Fix image preview reset logic
Add search parameter to createQuery function
Introduce configurable symbol name for ComponentJS
Emit 'disconnected' event with additional info
Fix: Use `req.headers` instead of `req.header` for `accept-encoding`
Use `nativeClass` for array type check
Update CLI command to use -u flag for webhook forwarding
Refactor truncate filter for improved logic and clarity
Remove console logs from conflict tests
Refactor: Resolve filepaths in `it` task
Refactor: Move Users instantiation into a Proxy
Fix: Set publicPath in dev config for correct image loading
Fix: Correctly resize deadzone on camera update
Update LineChart component to handle data changes more efficiently
Refactor: Custom CSS loading logic and add family support
Update code font family
Refactor: Use 'style' instead of 'styles' in package.json
Refine Github API search query and format response
Refactor: Improve clarity in rdns_access function
Add type checking for props in CellCreatorButtons
Use `Boolean()` instead of `!!` for ActiveXObject check
Refactor sunburst chart column calculation
Update PromiseEvent timeout error message
Fix handling of query messages in getCommandString
Fix: Handle click events on non-actionable buttons
Add Saraiki language support
Log global and daemon errors
Use process.cwd() for checkbuild file path
Fix: Set aria-labelledby only if association exists in metadata
Add button states to theming system
Add test for ignoring clicks with meta key pressed
Refactor: Use fixed resource prefixes instead of dynamic ones
Extend Client options with config.extend()
Rename `jQuery.CSV2Array` to `jQuery.CSVEntry2Array`
Update JSLint status bar indicator with disabled message
Add stack tracking for failed require loads
Parse package descriptions as markdown (#115)
Refactor: Use dedicated function to remove process listeners
Refactor: Move _applyChanges logic and introduce afterDocumentChange
Fix: Handle standalone routes correctly
Fix charset handling in MySQL client
Fix console log aggregation for multi-run scenarios
Refactor: Remove unused deep promise check
Refactor: Move `getConfig` function call inside `deployAll`
Refactor: Handle skjar1 and skjareinn routes with a single endpoint
Add support for matching on HTTPS port in URL filters
Refactor: Use 'type' instead of 'className' for property detection
Add tests for range parsing with large integers and max safe integer.
Fix: Handle grid errors properly
Add onEnterKeyPress prop and default function
Fix: Navigate back after ignoring a folder
Add translation functionality for Label model
Fix incorrect resource path in order test
Refactor: TimelineSignal Value Handling
Add findOne function to mongodb service
Pass event object to onFocus, onBlur, and onChange callbacks
Refactor: Improve worker shutdown signaling
Refactor: Hide trees and actions sections on tab selection
Upgrade phantomjs dependency to use prebuilt binaries
Add support for callback function with smartFlatColor
Add Mixed type to schema definitions
Fix: Handle `null` values for `params.term` in delay logic
Refine schema form key definitions for attributes and shoulders fields
Add memcached startup interval option
Refactor: Set environment to 'dev' in watch mode and adjust webpack configuration accordingly.
Trigger content_clicked event and close ActionBar on click
Fix: Pass index to `from` prop in useTransition
Remove console error logging in getTransactions method
Update Stamen tile URL for faster delivery
Use outputPath for permalink mapping
Add test for syncing a document between pouch and couchdb
Update SnackbarContent background color calculation
Add placeholderTextColor prop and type
Prevent pagination if tabs are not loaded or noPagination is set.
Add logging functionality with defined levels
Introduce DeprecatedContainerView for legacy support
Move logging import into wdSession constructor
Remove jquery-catchpaste dependency
Add `isForced` function to check if a path is forced
Refactor: Use template literals for endpoint URL construction
Fix requestAnimationFrame polyfill binding
Add test for put with _deleted: true
Update FilterRepositoriesController to display repository count
Update icon classes for note tags
Increase priority of Decisions view
Refactor: Remove error handling and simplify server setup
Update module export string
Force load of UV by setting `embedScriptIncluded` to false (#2906)
Export easyComp and easyStore using default export
Refactor ProjectManager: Simplify directory change handling
Refactor: Improve offset() test tolerance for Chrome (#7730, #7885)
Refactor: Only toggle search menu on '/' key press outside input fields
Fix key name handling for Chrome and Safari
Fix admin filter logic
Remove dependency on random_challenge_generator.js
Use a more robust unique ID generation for testrunner.
Refactor: Resolve loaders and plugins for webpack config
Restore connectSid assignment in githubLogin
Update navigation logic to use 'link' instead of 'oldLink'
Refactor: Move Input stories to 'Forms' category
Add functions to modify raw theme properties (fontFamily and zIndex)
Fix URL decoding in StaticServlet
Enable debug functionality in browser bundle
Refactor: Remove unnecessary timestamp from zip log
Fix: Correctly match index file paths in JSX class name namespace rule
Refactor: Remove redundant chatroom query test
Refine wording in test case for Comlink function exposure
Switch webidl2 import based on JSCOV environment variable
Only render discussion list if navigation is hidden
Add significantPrecision option for Bitfinex exchange
Handle async and sync parsing with error handling improvements
Refactor: Use strict mode and class syntax for MultiWatching
Fix Mine link format
Fix: Test for removing styles now correctly checks for absence of 'top' property
Use os.tmpdir() for locale path
Log errors from `meta.find` with more context
Fix: Correctly merge configuration values with objects
Exposing TAP instance as `tap`, `tap.t`, and `tap.default` for better usability
Add inclusivity option to between validator for date schema
Increase stdout buffer size for hermione process
Use Git.DiffLocation for diff URL instead of hardcoded path
Add dash pattern support for trace lines
Fix: Prevent input field from triggering pagination on every keypress
Pass `sorted` flag to Keyframes constructor
Fix: Cancel UR not called in TicketingFailed scenario
Use const instead of var for Ganache and assert
Improve error handling in task runner
Remove deprecation warnings for createPath and createHref
Fix recursive call in generate function
Fix: Properly parse repository details from Bitbucket API
Add tests for custom domain validation and usage

Move 'type', 'max', and 'min' attributes to the beginning of the list
Handle headers in request correctly
Update publicResolver address
Add tests for default migrations directory behavior
Add hover functionality and styling for color picker
Add arguments for xvfb and docker support
Refactor: Use canihaz for lazy loading dependencies
Add timeout support to A* pathfinding
Remove redundant export for Twig
Refactor: Improve is.equal function for type checking
Use HTTPS for Elasticsearch connection.
Refactor: Handle private routes in protectedRoutes array
Add tests for merge behavior
Add proxy address to getCdps schema ID
Refactor: Rename `tst` to `currTest` for clarity
Refactor: Check for Vue instance in unbind function
Fix: Handle bugos resize value in image processing
Refactor: Simplify flow starting logic in NodeTestHelper
Fix tilelive.auto() protocol handling
Refactor: Replace MXFocusTrap with RestrictFocusToChildren in Drawer component
Add debugging functionality
Add battery source and save device info for Orvibo devices
Add self reference for chat message function
Refactor: Simplify graph.js by removing redundant `attrs` function
Add support for extra packages in jam configuration
Enforce existence of specified log file
Fix: Call db.clear with callback
Refactor: Use API version in Stripe initialization
Refactor Account.setCode to remove compiled flag and simplify logic
Refactor PDF printing setup for accuracy
Fix: Remove trailing slash from image size path
Refactor: Use LazyFiltersResult for ramp calculation
Add deltaX and deltaY to mouse update events
Add `shouldUpdate` prop type to NavigationBar
Refactor: Simplify `set` method in Deserializer
Add service spec file if clea.spec.service is defined
Add debugger statement and set paused flag in playTestOrResume method
Refactor: Shorten import paths
Update electron task for arm architecture to use armv7l
Refactor: Optimize slide index calculation
Add wait for notification dismissal in specialCharListLinks test
Allow closing dialog with Escape key by executing default action
Add playback rate control and retrieval functions
Added UserDumbComponent and integrated it into App
Add full screen prop to Modal API documentation
Add support for Mosaic Layers in ArcGisCatalogGroup.
Rename Tessel using MAC address instead of fake name
Use arrow functions for selector evaluation
Add old password prompt for user account update
Add removeFile method to NodeStatics
Refactor: Change padding to margin in Item component
Handle SC2Map paths differently in createEngine function
Fix unexpected-messy test case handling of Buffer objects.
Fix: Remove bogus HTML from block response parsing
Fix: NW.Dom - Correctly detect buggy getElementsByName behavior
Improve LiveReload performance by only writing annotations file when changes occur
Use ReactDOM for rendering and unmounting components within the frame
Mock CUI Modal for editor tests
Fix: Update Catalan translations for timeAgo directive
Add tests for charset handling
Fix: Use template literals for comment loader ID
Add test for custom error handling
Stop drones with error handling for monitor stop
Update AccountController to use Model instead of model
Fix: Improve template resolution error handling
Fix: Log only the first item in the buffer
Ensure Nginx start callback is only triggered once
Fix: Swap row and column indices in wellKeyToXYList
Prevent empty changes from triggering history updates
Fix: Handle errors from Superagent responses
Add output for ImageFade node
Fix: Avoid duplicate paths in CSS source extraction
Refactor: Handle Elasticsearch index errors with more context
Add event listeners for property name and db name changes
Add toggle UI option for schema widgets
Log button visibility and handle potential errors
Fix: Remove unnecessary comment in rightBrace method
Configure: Use 'auto' prefix for exposed tasks by default
Refactor: Simplify topic handling in multiCheck
Add native property to realpath and stat functions
Remove unused dependencies streamify and vinyl
Handle save errors and display user-friendly messages
Refactor: Pass props correctly to nested components
Refactor: Set values.settings.facade() instead of global.SETTINGS
Disable CSS URL processing in Sass compilation
Only install GitHub workflows if .github directory exists
Remove default focus from first input field.
Add function to parse cookies from request headers
Fix: Clear spinner timeout even when not submitting
Fix DOM readiness check for non-browser environments
Adjust grid height on template updates
Refactor: Calculate modal max height based on window size and content.
Fix: Date constructor remapping for years 0-99
Use availableOptions for initial whitelist instead of an empty array
Add lodash dependency for QueryEditor component
Refactor: Simplify value handling in VTextField
Fix: Correct regex in path splitting
Refactor notification polling logic for focus/blur events
Update ad script with client ID
Refactor: Use THREE namespace for Vector3
Add rightHeader prop to Card component
Use async/await for youtube search API call
Add support for providing a custom promise implementation
Add processor for packet type '103'
Fix: Prevent fake stop on click events
Add blur event listener for ContentEditableBinding
Add timestamp and datetime fields to aofex response
Add file length and md5 to readStream response
Reduce profiler buffer size to 10MB
Refactor: Structure index handling
Fix: Handle empty function names correctly
Fix: Ensure v-click-outside works only in browser environments
Fix viewport size calculation when container is empty
Add support for ImportDirective and StructDefinition in ContractSource analysis
Ensure webview content window has focus on initialization
Update play modes list to include SHUFFLE_NOREPEAT
Ensure activity data is present before mapping in flag plugin.
Pass 'NA' for environment in copy constructor if detection is unreliable.
Ensure ES6 Set and Map are available
Use unique ID for each stomp client instance
Fix string detection in stream function
Add syntax error handling and colorized output for test_filter.js
Export htmlParser as elenpi.htmlParser
Add error handling and finish event listener for output stream
Refactor migration logic to handle different database types
Expose `readers` and `writers` as module exports
Enforce valid semantic versioning in CHANGELOG generation
Refine description of authentication source association
Improve `render` helper by explicitly checking for controller name and asserting its existence.
Reduce minimum input length for XREF/MREF search field to 1
Add 'Column', 'SlimBanner' to site index
Set locationType to 'hash'
Fix calculation of 'end' parameter for historical candles data
Refactor: Handle TCL comments more efficiently
Refactor: Move BlockchainWallet constructor to a function
Fix position property for IE and Firefox (#123)
Update getArtifactMtimes to resolve distribution directory relative to process.cwd()
Remove unused 'output' config option from background.js
Update npm registry check for latest plugin version
Use `requestAnimationFrame` for asynchronous execution
Remove HTML comment stripping from content processing
Fix timestamp display by accounting for timezone offset
Refactor SVG Sprite Loader config handling
Add return value for createVao method in GeometrySystem
Use global document for Delegator when no document is provided and update cache key generation.
Add pastebin feature and remove code execution instructions for non-EVALEX environments
Remove dead coroutines
Use 'new' keyword when instantiating Chance object
Add included hook and register version script
Add check for `cloneNode` behavior with checked attributes
Refactor: Improve frameDebounce with cancel functionality
Refactor PIDController initialization
Improve web push notification sending
Fix: Provide Date.now() fallback for older browsers
Disable sync loader and enable combo API
Protect vTree after attribute changes
Fix: Ensure HTML input is a string before processing
Fix: Handle HTTP errors in EnsemblFeatureSource fetch
Fix dropColumn logic for SQLite3
Set pixel ratio for canvas rendering
Copy htmlState in htmlmixed mode
