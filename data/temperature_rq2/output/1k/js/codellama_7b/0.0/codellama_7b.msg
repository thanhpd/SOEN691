Update track color property to use default color if not specified in config. Fixes #1234.
Refactored connection manager to improve performance by reducing unnecessary task processing
Added button type for begin and end buttons in ResponsivePopover control
Update hand-controls component to use LoopRepeat instead of PingPong for gesture repetition
Add support for constants in rollup.iife.config.js
Fix error in ${reportName} report with end date before start date
Added support for Node.js exit code handling in the host module
Fix issue with destroying generic role in afterAll hook
Update third-party library mapping in test/third-party.js
Added support for template error handling in webfont task
Refactor: Improve performance of findModelNodes function by reducing unnecessary iterations and using a more efficient algorithm.
Refactor input event handling for IE8-9 compatibility
Update AccelerometerProxy for Windows 8 platform
Update normalizeEntrypoints function to support Webpack 4's new entrypoint format
Added error handling for missing keys in IFrameApi class
Update require path for ui-layout module
Added a reference to the validation messages for each field in the form. This allows us to keep track of the validation messages for each field and display them appropriately.
Update CLI configuration file for new OS support
Added support for uploading files with a unique identifier and created a new directory for each file using `mkdirp`.
Update peity grapher to use each() instead of change()
Fix issue with iframe not reloading after template change
Update trace.js to use original callSite instead of slice() for better error handling
Update modal animation type to 'fade'
Fix issue with adding old MS filter for IE8 and IE9. Fixes #1234.
Added support for enter key press on select component to trigger save action in controller.
Update sort options for datatable widget
Added data-id attribute to regions for easier identification
Added support for custom command options using program.option() in ShellCraft.prototype.cli()
Update IconButton component to support SvgIcon as well as Icon.
Fix issue with transformPosition method when source is USER
Refactored selectors optimizer to use new SelectorsOptimizer class. Removed unnecessary options and added compatibility with new UrlRebase class.
Update TextMetrics class to support overwriting of text styles and improve performance by reducing unnecessary re-rendering of text.
Update live tab selection on pointer move and up events
Added unit tests for checking if package contents exist
Update push notification payload validation to ensure it is an object
Added support for searching agents by UUID or address in data.js
Update network update frequency and add lastX and lastY variables for better performance
Fix issue with executing scripts in non-interactive mode. Fixes #1234.
Update current running test export
Update script for running Webpack with production configuration
Update watch options for Windows users
Update `addComponents` function to `addUtilities` in `src/index.js` file.
Refactored GameEngine class to use a new physics engine implementation. Removed deprecated code for initializing the physics engine.
Update queue configuration for BullMQ with new Redis options
Update options for util.inspect() function to include depth and colored output
Update bundleWebpack() options to include a new configPath parameter for specifying the path of the Webpack configuration file.
Fast reconnect feature for IRC client library
Fix issue with firing change event on touchend when dragging handle. Fixes #<ISSUE_NUMBER>. (#<PR_NUMBER>)
Update custom marker directive to set custom marker on map initialization
Update icon family for Ionicons icons
Update jest config file to use ts-jest preset and remove rootDir property
Fix issue with server pool not being destroyed correctly when closing a server connection.
Adjust slideshow height based on image proportion and parent element's height.
Add support for SmartThings Z-SRN12N smart siren
Update publish flow options modal to support new email and scheduled publishing features
Added support for setting the ancestor property of children and added a new method to sort children by their z value.
Update CLI utilities for Windows platforms to use Unicode symbols for OK and error indicators, as well as add new symbols for folder and root icons.
Update observable.js to use toString() instead of toString(options) for better compatibility with older browsers.
Added support for reading extended header frames and updated the frame reader to consume the bytes after successful parsing.
Update config.js to support environment variables in theme and plugin directories
Added sudo command to getDefaultInterface function to get default interface on Linux systems.
Refactor Redback constructor to accept a Redis client object instead of individual options for command_queue and offline_queue.
Added support for disabling and enabling the link button in the editor toolbar based on the selected node.
Added support for renegotiating transceivers and requesting negotiation from initiator during ICE candidate gathering process.
Update edit form controller to use new validation object property
Update test specs for LowLevelFileIO and CSSInlineEditor
Update support for pointer events and touch input detection
Added support for new event type in API
Update test_modrewrite.js for grunt-contrib-connect-modrewrite
Update uncss task to support multiple files
Update Chance.js to use new Date().getTime() for seeding randomness
Update marked.js to fix a bug in the paragraph tokenizer
Refactor streamCompare function to accept options or comparison function as a parameter
Fix broken MotionGuidePlugin data
Update syncer test to use new backboneSync signature
Refactor text component to improve performance and fix highlighting issue. Removed unnecessary lines of code and added new styles for highlighted text.
Prevent click events from non-keyboard sources and add a flag to identify Ionic app events
Fix context utils to handle empty vnodes correctly
Added support for queueing connections and added selectors for retrieving the queue.
Update current step in WizardProgressNavigator control. Fixes #<ISSUE_NUMBER>. (#<PR_NUMBER>)
Update dynamic table functionality to support nested tables and lists
Update typography theme to use a new font size and line height based on design requirements.
Update resolved path for Generator class
Update SiftScience script loading mechanism to use const instead of let for better performance and readability.
Added support for a `before` option in the constructor of VerifyStream, which allows users to specify a custom stream that will be piped into the gunzip stream before it is used to parse the tarball data. This allows users to perform additional checks or transformations on the data before it is passed to the TarBuffer.
Update karma-minified.conf.js to exclude smart.elements.js instead of smart-elements.js
Update dependency on level2/core.js to use new module name
Fix text rendering when no fill or stroke style is set.
Update cache entry mapping to include whitelisted properties
Fix server error handling in NylasConnection class
Added `getLanguageId` method to `NullLanguageMode` class to support language-specific highlighting.
Fix warning for missing dependencies in node_modules folder
Added IAM service as a dependency for CorrelationService registration in order to support authorization checks during correlation creation and deletion.
Update strategy for VKontakte authorization with support for custom scope
Fix error handling in sendMessageProto() function
Update text styles to use new color and letter spacing variants
Update uWebSockets version in details script
Update cooldown timer initialization and add support for multiple spaces in command matching regex
Added support for capturing stack trace in StatusCodeError constructor using Error.captureStackTrace() method.
Added test for StyledComponent creation order and fixed CSS ordering issue
Added support for tracking events with visit and visitor tokens
Improve performance of resolveRef function by optimizing index ref lookup
Update withDefault-spec.js to use new default value for testing
Update last message in conversations model to include description instead of body
Update HTTPS agent for secure connections
Update highlighting function to use ignoreIllegals option in hljs.highlight() to prevent errors when rendering code blocks with unknown languages.
Fix issue with drag and drop file uploads on webkit browsers
Update Bundler test to use new ProgressBar API
Update test.js to use waitFor instead of immediate for better test stability
Fix file path for remote files on non-blackberry10 platforms
Update dependency on index.js
Added Thai language support for the designer
Update statistics calculation to include number of values
Fix issue #6574 by ignoring `__ob__` key in `inject` object.
Update interact.js with new features and bug fixes
Update lint-bare-strings.js to allow for additional characters in bare strings
Added support for retryable reads in executeWithServerSelection function
Refactor ArrayBuffer conversion to use Buffer.from() instead of creating a new buffer and copying data manually.
Improved performance of WebSocketTracker by limiting the number of offers generated and using Math.min() instead of Math.max() to ensure that numwant is always positive.
Update rollup config for ESM output
Fix typo in `baseVolume` and `quoteVolume` properties of `ticker` object in `gateio.js` file.
Added support for filtering by nested fields in query helper
Update featured playlists controller to include ISO 8601 timestamp in request
Update withdrawal requests API endpoint to use milliseconds instead of Date object for timedelta parameter.
Update test modules and add console logs for debugging purposes.
Notify watchers of new AA request
Added support for picking specific fields from log messages using lodash's `pick` method
Added two new benchmark fixtures to the suite for project 219313833 with warm-up time of 0 and recording time of 5000. Also added a results view for the suite.
Added support for rimraf to clean up temporary files after generating docs
Improve error formatting by trimming long lines and adding ellipses when necessary
Fix for issue #1234 - Quitting the application when in production mode
Added touch event listener for single-finger taps on document element. Prevents default behavior and prevents scrolling when a single tap is detected.
Added support for parsing multiple key-value pairs for a command line option using a custom map function. This allows users to specify multiple root password options in the form of old:oldRootPassword,new:newRootPassword.
Fix overview map control destroy method to prevent double-removal of DOM elements
Update Jetpack Connect routes to use new endpoint prefix '/jetpack/connect'.
Update cache key generation function to use crypto module instead of fs and path modules
Fix incorrect assertion for bid and ask prices in testOrderBook function
Added validation for custom props in validAttr utility function. Added test cases for invalid and valid custom props.
Added support for multi-line comments in jstransform function
Update API middleware to use final adapter for request and response handling
Update nconf-redis-test.js
Fix issue with listBrokers method not emitting error event correctly
Update globals list in Runner constructor to include 'errno' global variable
Refactor SVN library node to inherit from ScmLibraryNode
Deploy Mock Oracle contracts for testing
Improve performance of getCoordsFromPoint by reducing the complexity of the search algorithm
Added support for object-assign library to merge objects in actions.js file
Fix webview preload script to handle new document type and outerHTML property correctly
Update CKEditor configuration for template tags
Refactor traverse function to handle frozen objects
Fix issue with missing attributes in analytics.js
Update package manager script to use yargs-parser instead of minimist
Added support for loading files without a protocol
Removed unused exchange 'zb' from ccxt.js
Update entity data for blocks
Update MJMLColumnElement component to support new padding attribute
Update camera class to handle new GetScopesResponse format
Added a new method '_logPeerStatistics' to log peer statistics on a repeated interval.
Added support for retrieving remote address and port information from TCP streams. Added a new `rinfo` object to encapsulate this information.
Update Redux type definitions to use $Shape instead of | void for state in reducer function
Added support for copying information from a spec to a type definition. This includes copying the span of the spec if it is not already set on the type.
Update dependencies versions for canary and beta releases
Added support for touch events in wordcloud2.js
Update i18n strings for English (US) language
Update StepGroup component to support unstackable prop
Refactor highlighter plugin to improve performance and fix a bug with multiple selections.
Update require function to use path.join instead of path.normalize
Update Karma configuration to use ChromeHeadless browser with custom Chrome binary path
Added support for making paths relative to a context directory in the `makePathsRelative` function. This change allows for more efficient and accurate path normalization.
Refactor buildHashTimeLockContract function to return a more concise object structure
Refactored Select Directive to improve performance and accessibility
Update base64 encoding and decoding functions to use window.console instead of console object for logging.
Update skip links to use new event listener API
Update API client to use preacher-api for logs
Update last update time for server in replset topology
Update configuration file generation logic to use new solhint:default preset
Fix incorrect logic for determining active routes in `isActive` function. Simplify the code by removing unnecessary checks and using a simpler comparison.
Update fetchLedger method to require a code parameter
Refactor GuildDeleteAction to delete channels and voice connections before deleting the guild.
Update viewer and navigator display placement in NavigatorOnJQueryDialog test case
Update xml mode to support multiline tag indent past tag option
Fix issue with connection not being closed before connecting
Added support for deleted messages in Message class
Added support for initial input type
Update ESLint configuration to disable node/no-extraneous-import and enable node/no-unsupported-features/es-syntax and node/no-unsupported-features/node-builtins
Added support for timeInForce parameter in placeOrder method
Refactor function-uncurry-this module to use bind and call methods instead of apply method.
Added support for configuring browser-index with a configuration file
Refactor Flickr API query to use new MongoDB driver
Update getLibraryPath function to use new URL format
Added support for loading documents using a custom document loader function.
Update regex for matching links in markdown and HTML to handle escaped characters correctly
Refactor triggerKey function to handleEvent for better performance and flexibility
Update RDF export form UI to improve layout and functionality
Update task.js template with new logic
Update line width based on camera upsampling factor
Update fixture element with new content
[Arrays] Update indexOf() method to use native implementation
Added support for custom linker options in Gruntfile.js
Fix issue with setting position of tab to 0
Update SimpleFormIterator component to handle empty fields prop
Refactor: Simplify isZeitWorld function by using a Set and an Array method instead of a custom implementation.
Added support for Google Calendar ID detection in V1 and V3 API feeds. Will now detect calendars like 'asdf1234@calendar.google.com' in addition to person email calendars.
Fix virtual scroll offset calculation for correct scrolling behavior
Fix bug in loadRolesIfNeeded function by updating setPendingRoles call to use dispatch and getState parameters correctly.
Update modules list and display updated modules in console log
Update default theme font sizes and weights
Added gain and pan properties to Sampler class for easy control of sample playback.
Optimize stack trace caching for better performance and memory usage
Update settings getter and setter functions to support default values
Update config file path in spec tests
Added option to disable saving of associations when saving an instance. This allows for more control over the order in which associations are saved.
Added support for retrying failed tests and suites. Added new actions (RUN_FAILED_TESTS, RETRY_SUITE, RETRY_TEST) and updated reducer to handle these actions.
Added support for sistyl instances in .set() method. (#1234)
Update gulpfile.js to use new TSLint API and fix watch task
Improve stack trace formatting for better readability and accessibility
Update URL parsing and navigation logic
Added lunr.utils.asString function to convert an object to a string.
Update opacity property of ImageryLayerCatalogItem to use Number instead of String.
Update copyright information in Docusaurus config file
Update filebrowser service to handle CWD deletion and selection updates
Update GuildChannel structure to include nsfw property
Fix postcss-modules plugin options for scoped CSS classes
Fix issue with Firefox cache miss causing empty file uploads
Improve sequential navigation heuristic for finding first focusable element in composed tree. This change addresses an edge case where the browser's algorithm may not work correctly.
Update device socket ID and online status in database for socket with UUID <UUID>. If device is not found or token is not valid, return error.
Added support for new inspector editor identifier convention in UI extensions
Update range.js library with new features and improvements
Refactor: Remove unnecessary code for development mode check. (#1234)
Update accessibility for fragments announcement in screen reader mode
Fix issue with eliding type-only imports in Babel plugin for TypeScript
Added timeout for client requests to prevent hanging connections
Added support for asynchronous specs
Fix issue with transferring data between frames in admin panel
Update event listeners to use EventTarget prototype methods instead of DOM Element prototype methods
Update timestamp function to use performance.now() if available
Fix IE7 focus issue in SlickGrid. Fixes #1234.
Added transaction ID generation for messages sent over websockets
Added support for long stack traces in Promise code.
Allow moving nodes in the tree view only if they are not archived and inside the root container.
Update gulpfile.hygiene.js with new file pattern for snapcraft.yaml
Refactor notification component to use new test utils
Remove unused code for domain renderer function
Added support for sending cookies with HTTP requests
Update chainable model to clone internal attributes and clear relation_to on clone
Added support for 1m, 5m, 15m, 4h, and 1w timeframes in hollaex exchange
Deprecate `fired` in favor of `hasFired` and add support for pending callbacks
Create Solr cluster with name `example_cluster` and size `1`
Update channel mapping file path in getChannelMapping function
Update computed properties in View class
Refactor EventListenerView to format code for better readability
Update Pusher unit tests to include new cluster parameter and fix warnings for missing cluster parameter
Fix bug in query options close button
Added support for bearings in route API request
Refactor movies parser to improve movie name parsing and remove unnecessary parts
Added support for new condition types and operators
Added support for proxy configuration in WooCommerceAPI constructor options.
Update public key algorithm information for MPI-based keys
Fix stream error handling for WebSocket close codes
Added support for sanitizing page URLs in Construct function
Update percentage hook to use reference window for calculation
Refactor code to use async/await syntax for better performance and readability
Refactor: improve performance of `ensureBlock` and `toBlock` functions by reducing unnecessary array creation and checking for statement types.
Refactor history implementation to use a class instead of a function. Add support for navigation back and improve performance by using a singleton instance of History.
Improve performance of find.js by reducing the number of iterations in the while loop
Fix issue with dropdown trigger not working on non-left mouse clicks
Added support for implication symbols in parser
Improve performance of `nwmatcher` by reducing unnecessary object creation and string concatenation
Update Crosswalk version check to support versions up to 14.
Update schemaErrorReporting in .boltrc.js
Allow binding multiple listeners at once for `on` method
Fix non-Host handshake warning in Pipe.parseHandshake()
Refactor account component to use inux and handleActions for state management and effect handling
Added validation for `src` and `dest` properties in `buildFileList()` function of `catena.js` file.
Update backend adaptor to use new socket.io library and add hostname and listen port parameters for receiver.
Added support for initialFocus and ariaDescribedBy associations in ResponsivePopover control.
Activated user password for login test
Added support for seeking in HTML5 audio playbacks
Added support for renaming files and directories
Update ProfileService factory injection and implementation
Fix issue with getTagsRecursive function when passing an element as argument
Update MultiInput tokenizer width calculation to use scrollWidth instead of boundingClientRect width.
Refactor action function signature to support new options parameter first and promise parameter second.
Register custom schemes with privileges for VS Code webviews and resources
Added support for `target` attribute in `<resource>` tag of config.xml file.
Fix unbinding of click away event on component unmount. Previously, the event was not being properly unbound when the component was manually bound.
Update katello_object.js to use blur event instead of focusout for name input field
Increased crawler timeout to 500ms to handle longer requests.
Refactored file upload functionality to reset files list and clear previous files before sending new ones.
Refactored RenderMode.makeGLProgram() to improve performance by reducing unnecessary rebuilds of the same program.
Fix bug in local storage detection and add support for process.env.DEBUG
Update margin for barchart.js to improve layout.
Update index for torrent and emit event for seeders count update
Fix parsing of vCard files with multiple lines
Update fromEmissions function to sort emissions by their x values before mapping them to a new array.
Fix swipeable views issue with negative indexes and update startX on touch start
Refactor generate-icons function to use sharp library for resizing and flattening images. Add support for background color option.
Update mime type parsing for form encoding
Fix dropdown directive to prevent unnecessary re-rendering of options
Added support for decimal precision in masking function
Added support for loading raster textures from main thread and updated texture width/height uniforms in RasterStyle.
Update git fetch and clone commands to improve performance
Fix issue with adding tag file to Uppy instance during webcam capture. Previously, an error was thrown when trying to add the file, but now it is handled and the file is added successfully.
Fix global material IDs in normalizeMaterials function
Update ember-cli-build.js to use ember-native-dom-event-dispatcher in development mode only
Update dashboards table to use dashboard ID instead of name
Refactored CanvasImageSaver to use window.canvas2ImagePlugin.saveImageDataToLibrary instead of window.plugins.canvas2ImagePlugin.saveImageDataToLibrary
Update ui-slider component to support custom tooltips and disabled state
Update Bcrypt hashing algorithm to use configurable rounds
Refactor error handling in upload progress bar directive
Update karma configuration file to include new test files and reporter options
Added layout test for short cycle with ranksep set to 200
Update ParsoidService to use production environment by default.
Update maximum number of open files to 1024 and add a retry mechanism for EMFILE errors when opening files. This change improves the performance and stability of the application by reducing the likelihood of running out of file descriptors.
Added support for parsing arrays in MapTransformer.js
Update jQueryBuilder test file with placeholder for future tests.
Added support for deferred initialization of extensions (issue 1034)
Increased test repeat count and concurrency limit for more thorough testing of streamr client library.
Fix in-memory SQLite3 database connection pooling issue
Added Travis CI integration
Fix issue with encoding of field names in URL-encoded data. Previously, the decoder was not reset correctly, resulting in incorrect field names being emitted.
Update calendar time format to use 24-hour clock instead of 12-hour clock.
Added tests for FormControl size prop
Fix issue with determining correct port when $PORT is set and --port is not specified
Added focus() method to set input type to text when visibility is toggled and added a new event listener for togglePasswordVisibility button click.
Added non-minified substance file to test folder for testing purposes
Update escapeText function to use a div element instead of a p element for escaping text.
Added a stable iteration mechanism for IterableMechanism to prevent unstable ordering of keys and skipping keys during removal.
Refactoring of TMW.SiteSetup module to TMW.Config and adding an example module for UI interactions.
Update hash function to use data.data instead of [...data.data]
Added caching headers for improved performance and reduced load on the server. Set 'Cache-Control' header to 'max-age=50' and 'Expires' header to 50 seconds in the future.
Update parseHeaders function to handle string values correctly and remove unnecessary code.
Refactor: Remove unnecessary dependency on esprima and simplify import resolution logic
Update key path getter in Index class
Update test cases for number-zero-length-no-unit rule
Remove transform from collection prototype
Update i18next provider to support custom translation options
Update resource saving mechanism to include callback for updated resource
Added test case for route matching with a colon in the path
Added validation for missing parameters 'schema' and 'fn' in function 'validateAny'. Fixed bug where 'keyPath' was not being passed correctly.
Update HTML5Audio playback source and reset src when stop is called.
Refactor legacy interface code to remove unnecessary PayPal Checkout Ready functionality
Update damage calculation for creeps and structures to include dismantle attack type
Update input area styles for better padding and margin on mobile devices
Update polyfills for nub, union, and intersect functions
Added support for serializing Sets and Maps in IBase class
Added support for flushing the read buffer in stream/pipe
Added support for authenticating requests with a specific strategy and options. Added ability to pass in custom options and callback function.
Refactored client.js to improve performance and reduce memory usage by reducing unnecessary object creation and garbage collection.
Added afterEach hook to restore simple mocks
Added test case for `each` method returning an error if the connection dies
Added tests for changing foreground and background colors via string parameters in Cursor class.
Added support for passing response object as a context parameter in getEnhancedContext function
Update JWT verification for API requests
Fix plural forms validation regex
Added support for non-attribute options in `getAttributes` method of `BaseView`.
Added create contact button to email detail view
Fix console log statement in index.js
Fix issue with loading images in lightbox directive by using ImageLoader service instead of directly setting src attribute.
Added support for strict mode in Opal runtime.js file.
Refactor log class to use lodash's pickBy function instead of pick
Add support for passing a function as a selector in dq() method
Fix bug in handling StreetView position changes
Update README with new options for caseSensitive and defaultHandler
Added support for refreshing tabs when they are clicked and there are pending updates
Removed some relative replace statements from go.mod
Added self reference in Drone.prototype.list() method to improve performance by reducing the number of times the format function is called.
Fix memory leak in HTMLCanvasElement#getContext('2d')
Added invariant checks for key, action, and handler in socketsync function. This ensures that these parameters are not null or undefined before using them.
Update mnemonic generation to use 32 bytes of entropy for improved security
Update Slide component dimensions
Fix position and size of new windows created with BrowserWindow.openDevTools() and BrowserWindow.loadURL().
Added methods for getting view and data positions of annotations
Fix: Update test fixtures for has-many-dissociator service
Update import paths for @paypal/sdk-constants to use src folder
Added connection_found flag and triggered hooks.trigger('connection_found') when a connection is found.
Added 'wx' flag for creating a new file with exclusive write access
Added write and send methods to Socket prototype for sending messages over the socket connection.
Refactor on.js to improve performance and reduce memory usage by removing unnecessary code and optimizing the execution of event handlers.
Increase gas value for deployment of smart contracts by a random percentage
Update log function to include timestamp in log file name
Added stopPropagation() event listener to prevent indeterminate checkbox from triggering parent element's click event.
Remove is_in_app check for hoodie-server
Update layers in PropertyStack model
Refactor async-to-promise library to use es5-ext functions
Update icon component to support new fontawesome icons
Added support for creating directories for migrations and models using mkdirp module. Added config options for migrations-path and models-path.
Added filter for section in main controller
Fix bug in NavigationInstruction class by removing unnecessary property 'plan' and updating the constructor to set it to null by default.
Added support for passing rubicon-specific parameters in bid requests
Update difference and intersection functions to use new Set() for deduplication
Added support for RGB color mode in Yeelight devices
Added cache busting for IE / Android / iOS6 browsers and enabled timestamp requests by default.
Refactor find and replace functionality in orion/editor/find module
Added required prop 'apikey' and new prop 'searchInput' to component 'PlacesAutocomplete'. Also added a default value for 'type'.
Fix config file path for embark.json and embark.js files
Update cooldown exclusion check to include guild ID in the check for channel and user IDs.
Update license badge for vscodium-bin package
Added support for custom form data setup function in inline attachment plugin
Update search results for source records with new id and url properties
Fix issue with onQuit callback not being called when creating a new session.
Update Bootstrap version to 3.3.7
Update peer ID format in announce response for IPv6 peers to use binary representation instead of hexadecimal string.
Added support for debouncing calculation handler with delay option. (#1234)
Fix incorrect reference to firstElementChild in test for recursive inline partials
Added support for 3DES encryption
Update migrations directory path resolution logic to handle missing key in config file
Refactored AbstractEditor class to use new dependency injection system for file client and save handler.
Fix login modal error handling and improve performance by using null as a callback for the app.session.login method.
Fix error handling for Instagram API calls
Update ticker data structure to remove prevClose field
Improve performance by deferring HTML state saving until last possible moment.
Added ember-moment dependency and imported it in blueprints/ember-date-components/index.js
Update GMapsObject initialization with default coordinates for map center
Update Onyx version to 2.3.0-pre.11-dev
Update plugin globalization module import path and add .js extension for Node.js compatibility
Refactored uploadFiles method to use FormData instead of creating a new FormData object for each file.
Added support for handling HTTP and HTTP API events in Serverless Offline. Added a check to ensure that the function has a valid handler before adding it to the list of events.
Added DHT announce and get peers for secondary info hashes to improve finding joystream clients for specific info hashes.
Update hieroglyphy.js with new character mappings for 'f' and 'l'. Fixes #1234.
Added support for emitting custom events on WebSocket connections based on the URL of the incoming request
Fixes a bug in RcmEdit.saveAjaxAdminWindowUsingPost() by removing unnecessary code and updating the URL for the POST request.
Added support for mock-fs' readable/writable stream implementation. This change allows us to hook into the 'open' event before any of the other listeners and add the mapping between file descriptors and mount paths.
Update inputs spec to use Pivotal UI React's control-label component instead of form-group label.
Added logging for local IP address and device list in test-capture.js
Added support for storing arrays in storage adapter spec
Added keyboard dismissal functionality for search input on mobile devices
Remove support for `/leave` messages and replace with `peer:disconnected` event
Update asset revving functionality to support new option for generating Rails manifest file
Added support for connecting to Tessels over LAN using a new `LANConnection` class. Also added error handling and filtering of invalid connections.
Refactor router helper to use async/await syntax and return a promise instead of using callbacks.
Update supported transition properties and add cache for improved performance
Update getDetailURL method in IndexPage component
Added support for blank keycaps in on-screen keyboard layouts. This allows users to type spaces and other special characters using the keyboard.
Update calendar body component to use new date styles and improve performance. Fixes #123.
Added test cases for `toBuffer` method with target Buffer and offset parameters.
Update SVGO configuration for non-spritesheet SVGs
Refactor: Move sanitization logic into a separate method for better readability and maintainability.
Update test coverage threshold for unit tests to 99% (statements, branches, functions, lines)
Update webpack configuration for development environment to show more detailed information about module sizes and add maximum number of modules to display.
Fix generate link command for non-Electron environment
Detach ad manager from video element when destroying player instance.
Update user settings and user guild settings on shard initialization
Update Tick component to display values with a negative sign
Update service method directory path to use `path.join` instead of concatenating strings.
Update main.js to use latest jQuery version from Google CDN
Fix issue with getWorkspaceMeta function not returning workspaceInfos correctly. Fixes #1234.
Update types and add new exports for Parcel, Member, Name, Repository, and Bugs. Also update the import statements and export type declarations.
Update TextField component to support custom styling
Added support for mobile dragging with touch events
Update HLS playback position and DVR settings
Update entity labels for article metadata
Fix incorrect test case for `is.not` method
Update test for GeoJSON source with new category names
Fix issue with missing metadata block in source file
Added a new method `ancestorPerformanceRestrictions` to the `pagesCursor` module. This method allows for more efficient retrieval of ancestor pages.
Added RX mode information for CCU connections
Added getName and setName methods to Shape class.
Added Google Maps widget to Airbnb search page
Fix error handling for invalid directory path
Update LabelEditSpec test to use skip instead of only for testing
Fix CSS selector helper for TBODY element
Fix whitespace normalization in content.js
Update fetchTickers method to use new response format
Fix cache has method for modules that don't implement 'has'
Added token retrieval logic for SocketService
Added support for setting Secure flag on cookies when creating them. This is done by passing a new parameter `secure` with default value of true and checking if the current context is secure before adding the Secure flag to the cookie.
Update filterMatchMode documentation to reflect correct spelling of 'endsWith'.
Update roc-package-webpack-web dependency to ^1.0.0-alpha
Fix null pointer exception when sorting table by numeric columns
Added support for measuring polygon areas and improved the measurement string format.
Refactor config initialization prompts and remove unnecessary code
Update asset picker in boom.page.js
Added support for children in Textfield component
Added source property to task objects for storing function source code
Fix file watching in CI environment
Update panel list functionality to include validation
Added method 'monthsInYear' to retrieve the number of months in a year
[Update Asset Cards] Update selected asset card
Add support for focus and blur events in ReactBlessedComponent
Refactor getConfig function to improve performance and reduce memory usage by reducing unnecessary object creation and filtering out blacklisted keys.
Added logging for assertions in actions.js
Refactor key navigation functionality to improve performance and fix a bug with negative default position values
Fix unsupported hash key error in Property.Type component
Added meta tag for UTF-8 encoding in HTML output
Added support for passing a custom node to the constructor of BaseComponent.
Added a new method `transform()` to the IgnoreFormatter class to handle code transformation.
Refactor IntlTelInputApp component to improve performance and fix a bug in the country data processing logic.
Clean up database connections in multicopy test suite
[Uncompiled Mode] Override define values
Added support for parsing numeric values with trailing decimal points in the `getComparator` function.
Update fetch function for compatibility with node-fetch
Fix insertion of element into DOM when force is true
Refactoring of generator-utils module and added validation for package name
Fix Ropsten issue with balance check
Fix textContent getter and setter for elements to handle HTML entities and non-breaking spaces correctly.
Fix focus on first input in modal form
Update canary version for install script
Update text wrapping algorithm to use SVG attributes instead of client rectangles for better performance and accuracy.
Fix error handling for script execution
Fix issue with `error` and `listening` event listeners not being restored after calling `_listen2`. This fixes a bug in the test suite.
Fix issue with binary data being converted to string and causing errors
Fix helpers validation in engineScript.js
Update entry point prompt for multiple entries
Added support for middleware in route controller
Fix issue with ESM import in gulpfile.editor.js
Added clear method to RenderTexture class to allow for clearing of the texture buffer.
Update Gruntfile.js to use new task for compiling DMN files (#1234)
Update chooseInstancePopup.js to handle failures in JSON request and display error message
Fix writing-mode and direction properties in postprocessTopStyle() method
Added a new function to handle anchor scrolling on content ready event with priority 50.
Update default server name and port in config file
Fix non-101 status code handling in WebSocketClient.js
Refactor Markdown.toCode() to use Hawkejs's parseTemplateElements() and addPrintCommands() methods to improve performance and reduce memory usage.
Refactor input option to use pmore.trigger() for keyboard events
Update not fully loaded directories in file tree view model
Fix issue with hasOwnProperty check in LokiOps.js
Update external socket connector to support snippet mode for proxy connections
Fix pagination for negative page relative values
Added support for withdrawing `code` currency
Refactored Shape constructor to use destructuring assignment for default values of optional parameters.
Update webpack configuration to use react-native-web instead of navigation-react-native-web.js
Update webpack configuration to include stats option for better build performance and error reporting
Added default renderer with lowest possible priority for shapes and connections.
Added Firefox detection test case to parser.test.js
Update auth user info object to include Firebase token
Update select2.js to remove unnecessary class and hide dropdown mask when dropdown is closed
Update connection and node filters based on new filter settings
Refactor core.js to improve performance by reducing unnecessary DOM manipulation and memory usage.
Update media event triggering logic to include new bitrate information
Added closure-polyfill for compatibility with older browsers
Update model-patch filter function to allow fieldless resources
Update Kue UI client to use new version of kue-ui-client library
Update universal render function to support async and promise resolution for undefined response
Update latency threshold for topics test
Refactored REST API to support filtering by type name
Refactor compare subtest results controller to use new series format for highlighted revisions.
Added SQRT2 constant for anti-aliasing in SDF shader
Refactor View class to use exec method instead of done method for rendering views.
Update core.js file with new code changes for inArray method and Safari browser detection.
Added support for custom class names on the ToastNotificationList component. Added a new prop called className that allows users to specify a custom class name for the component.
Update FairPlay DRM identifier for Shaka Player assets
Update reducer for moving groups in edit pod store test
Update BrowserStack configuration for newer Chrome and Firefox versions
Update test-rosidl-message-generator.js to require all message classes and use std_msgs/msg/String.msg
Update example pages for Dustbin component
Update progressive rendering for stream chart
Fix launching editor in serve mode
Added support for adding method missing stubs to prototypes
Update database connection configuration for testing purposes
Removed NoErrorsPlugin from dev config
Update generate-externs.js to handle new symbols and namespaces
Fix for null message handling in consume method of Remit class
Update file-loader configuration for SVG and PNG files
Update funnies.js with new jokes and features
Refactor defaultRender function in options.js to improve performance by reducing unnecessary object creation and property assignment.
Added support for storing UUID in local storage and generating it if not present
Added support for overriding minzoom in copy command
Fix issue with DropButton component where it was not properly closing when clicking outside of the button. Added check for 'open' prop to prevent internal state from being changed.
Update device model K2RGBW01 to support color temperature and color xy.
Update customized field label in validateElement.js
Refactor testflow task to improve performance and reduce memory usage
Update thresholds for my_rate metric in es6sample.js
Added navigation options for ListScreen component
Added test for saving a model with a string id and checking if it exists.
Update Zip compression level to allow for negative values
Update event listeners for FormBuilder component to use 'removeComponent' instead of 'deleteComponent'
Added missing spaces inside expression parentheses for better readability and consistency with other code in the project. Fixes #1234.
Refactored SceneManager to improve performance and reduce memory usage by removing unnecessary code and optimizing the start and stop methods.
Update publish command to use super method for exiting test process
Update request library to use v3 API endpoint
Update mobile auto-enable behavior to support touch events

Update pubKeyHash calculation using RIPEMD-160 instead of SHA-256 and RMD-160
Fix event listener exception handling in EventTarget class
Fix issue with deleting relationships between nodes with id_ prefixes
Added support for headless browsers in karma.conf.js
Fix issue with nested schema validation and defaults
Update browser.js to extract ID from URL and remove file extension
Update tests for connected state machine in Pusher.js
Fix HMAC verification for non-UTF8 ciphertext
Update modal open state and toggle body class accordingly
Update manifest for asset modules in Vue SSR client plugin
Deprecate 'CanService' service import in favor of 'ability' function
Update integration tests for content module
Update parent class for schema nodes to use AbstractNode as default value.
Update to use new package name for OOP library
Update jQuery File Upload Plugin to version 5.0.3
Update Gruntfile.js to include new directives and views in dist folder
Update parseOrderStatus method to handle new response format for order status API call.
Update UI to enable clearing of annotations and fetch new annotations from server on page load.
Update nodemon configuration to ignore build destination directory
Added support for checking if a file is a file before resolving it as a file.
Update input coordinates to account for device pixel ratio scaling
Refactor computed property function arguments to use ES6 destructuring syntax
Refactor InputMask component to improve performance by removing unnecessary code and reducing the number of re-renders.
Added support for using SSL certificates from /etc/ssl/certs on Linux systems. Added a new configuration option `useEtcSslCerts` to enable this feature.
Update writeStats function to resolve e.g.:"C:\\" issue on windows
Update attribution for IGN WMTS layer to include a link to their website and a logo image.
Added support for non-integer values in folder size check
Added test cases for ignoring full embedded entities in SirenExtension
Fix bug in textView.js where ruler was not properly removed when a cell was deleted
Fix issue link in global nav menu
Added support for customizing the data and public paths in Hoodie
Added support for callbacks in keys method of memory adapter
Added support for transactions in wire protocol version 3.2 and above.
Added support for JSON responses in HTTP requests and response objects. Refactored the request function to use a new Response class for handling JSON responses.
Added support for passing arguments to script being executed by wires
Fix issue with master link not being established properly
Fix indexedDB data retrieval issue
Update Korean locale file for moment.js
Update mentions link to use user slug instead of username
Added visibility property to new and old elements in explode transition. This ensures that the new element is visible during the transition and the old element is hidden.
Update version number to 0.3.1
Update user stream API test to use new inbox and act properties
Fix formatting issue in complete function. Add check for n >= 10 instead of n > 10.
Added support for undefined values in amDateFormat filter and accepted a numeric unix timestamp (milliseconds since the epoch) as input.
Added support for swiping inside a Modal by setting `onStartShouldSetPanResponder` to true.
Remove unnecessary loop in destroy method of SlickGrid class
Fix issue with event handling in Ember's `EventDispatcher` component by checking for the existence of an action before invoking it.
Update clickAndHoverHelper.js with new waitFor function
Update entity cache initialization to use an empty array instead of a new Array() object.
Update $resourceProvider defaults for pnc-client resources module
release: version {{newVersion}}
Added support for enabling and disabling the call module based on the visibility of the field trip window. Also added support for constructing new calls when the call history is visible.
Added support for Packstation addresses
Update Babel configuration to ignore node modules in the current directory
Added support for embedding environmental layers in the map
Update live reload configuration for ember-cli-live-reload addon
Added error handling for mapper function in split() method
Refactor Fn::Sub to use JSON.parse() and object destructuring
Update date format for Korean locale
Refactored component state management to improve performance and reduce memory usage.
Added support for XML namespaces in WSDL objectToXML method
Added support for Shopify, Ecwid, and Gumroad store plugins in WordPress.com Business sites.
Update activity instance tree with new instances and transitions
Fix XSLT transformation issue by removing unnecessary tags
Improved zooming performance by reducing the number of calculations performed during each zoom step.
Define IUnknown interface and its implementation
Update Webpack configuration to use awesome-typescript-loader instead of @ngtools/webpack
Update test configuration for Travis CI
Update onLevelLoaded API documentation
Optimize dynamic data rendering by reducing unnecessary render calls and improving performance.
Added support for dragging buttons
Improve OTA flashing experience by adding a delay before sending start command when using --reboot option.
Added support for regex search in tags
Update Hibernate version to 5.4.22.Final in generators/generator-constants.js
Update ESLint configuration to use Prettier and turn off unnecessary rules. (#1234)
Add support for installing manners from source code using gzexe tool
Added support for onCancelled event in Firebase's Android SDK
Fixes issue where caller was expecting first space but there were multiple spaces in the array.
Added support for reading sub-directories using lodash
Update NavList component to use new node structure
Removed unnecessary export of `start` method from sync module. The `init` method can now be used as a replacement for `start`. Also removed unused `globalHandleList` method.
Added new route for list-inspections page
Update gzipStatic configuration for production environment
Fix display issue in reflow function
Update index file with minified icons
Added support for excluding static parents in scrollParent calculation
Update content type regex for textual content types to handle newlines and whitespace in the content type header
Added event handler for ignoring double click events on layer switcher backdrop labels
Added test for setting an object to null without throwing an error in the form reducer. This is a fix for a bug where changing an object to null would cause the .$form property to be lost.
Added support for parent and children methods in V class
Ported getDistributionType function from cartodb-postgresql scripts-available/CDB_DistType.sql
Update constructor to prevent incorrect instantiation of Funnel class
Refactor getParent and getSiblings functions to use bitwise operators instead of modulo operations for performance optimization.
Added support for calculating trade costs based on price and quantity
Removed unnecessary newline replacement in goog.crypt.stringToUtf8ByteArray function.
Fix issue with pointer movement detection. The code was updated to require more than one pixel of movement before triggering an event.
Update Sass compilation directory structure for better performance and compatibility with newer versions of Node.js.
Added support for viewing slide notes in a separate window using WebSockets. The new feature allows users to view and edit notes for each slide independently, and also provides a way to share notes with others in real-time.
Added support for data-* attributes in dna.js core library
Added support for building from a string using `buildStr` method
Refactor getAliases function to use lodash map method
Added a check for whether the user can reply to the discussion before inserting a mention in the post-reply action component.
Fix: Remove unnecessary TinyMCE command to set content on init. This change improves performance and reduces the risk of conflicts with other plugins.
Added onOpen() callback for XHR transport
Update parseTicker() method to handle missing last_updated field and use milliseconds() function instead of safeInteger() for timestamp calculation.
Update Modal component to use new onClose prop instead of onBlockerClicked
Added support for date selection and formatting
Refactor PubSub.js to support context and method for subscriptions
Fix Gumshoe failure handling by checking for empty detection rules results and returning early with a null result.
Added timeout for PhantomJS process to prevent hanging
Added support for parsing properties with a prefix in the `JavascriptType` class
Added prorogram reference to options and commands for better organization and flexibility in the code.
Update browser augmentation to support HTML5 doctype
Fix: Update activities.js to use a stub for getToken()
Added support for removing multiple files from a form using an array of objects
Refactor HeaderSetCompressor to use a new HeaderTable class for storing and managing header entries. This change improves the performance of the compressor by reducing the number of iterations required to find matching headers.
Fix source map generation for require hook
Update module dependencies and add lodash for isSafeNaturalNumber function
Refactor PageItem component to use function components and improve accessibility. (#1234)
Added support for iOS app authentication using Safari and In-App Browser plugins. Also added a check for iOS app detection.
Fix list item indent lint rule to handle new `loose` and `spread` properties in remark-parse v6.0.0
Added timeout for Jasmine tests
Update Lines.js to support curves_subset and line_style changes
Fix parent ID and type for embedded documents in EmbeddedCollection.prototype.push() and EmbeddedCollection.prototype.add()
Update Set polyfill tests and add new tests for Symbol.iterator and Symbol.toStringTag
Added support for Bootstrap popover and tooltip handlers in project home view
Update mendelifyTransformStream to handle dependencies with avoidMendelify() function
Added test for default style language and disabled PostCSS by default in react-webpack:app generator
Added support for canvas element in web platform backend
Refactor: Replace usage of `sha3` with `keccak256` in security rules
Update widget.treeselector.js to use parent node ID instead of text for value
Update readLastUnit function to handle light clients with no last units in the database.
Update KilikTable.js with new features and bug fixes
Refactored autocomplete functionality to improve performance and reduce memory usage. Removed limitInCache option as it is no longer needed.
Added Util module for excluding props and added support for customizing buttons in Confirm component
Fix image preview directive to handle undefined image model correctly
Added new function createQuery in jquery.molgenis.xrefsearch.js
Added support for configuring symbol name via HTML tag <script data-symbol="<name>". (#1234)
Fix logoff event emitter to include error message from server
Update expressStaticGzipMiddleware function to use req.headers['accept-encoding'] instead of req.header('accept-encoding')
Update is-array module to use native-class for improved performance.
Update Smee CLI command and add usage info for Node.js client
Refactor truncate filter to handle empty values and improve performance
Fix conflicts in test.conflicts.js
Update ESLint configuration and add support for custom reporter
Refactor users module to use proxy instead of direct instantiation
Update webpack.dev.config.js to force Chrome and Firefox to load images from the correct path
Update deadzone size and position in Camera class
Update data and fill area in state when props change
Update custom CSS loading mechanism to support new configuration options 'urls' and 'families'. Fixes #1234.
Update font family for code blocks to use Inconsolata instead of Consolas
Update import logic to support new "style" property in package.json
Added support for searching by username in AsyncTypeahead component
Added support for IPv6 addresses in rDNS resolution
Added flow type annotations for CellCreatorButtons component props and context types.
Fix issue with detecting Flash plugin on Windows using ActiveXObject
Update sunburst chart layout for better performance and responsiveness
Improved timeout handling in PromiseEvent function
Fix issue with command parsing for queries
Added support for actionable buttons with new `isActionable` prop and updated `directives()` method to handle click events on actionable buttons.
Added support for Serbian (Latin) and Slovak languages with RTL directionality.
Update error handling in config reducer
Update check-build file path in check-build script
Update FieldBase to support ARIA labeling of content controls
Added support for button state in themer.js
Added test for ignoring clicks when meta key is pressed in main.spec.js
Update config file for Lambda Tools helper
Added support for overriding SQS queue URL in tests
Refactoring: Renamed jQuery.CSV2Array() to jQuery.CSVEntry2Array() and added a new function for converting a CSV string to an array of objects.
[JSLint] Disabled JSLint for this project
Added support for tracking the document stack when loading resources
Resolve and configuration file path, parse package description as markdown (as per #115)
Refactored worker test to use a custom removeProcessListeners function to clean up listeners after each test run.
Added afterDocumentChange method to handle changes from the collaborator's document.
Fix path handling in Controller constructor
Fix charset issue in MySQL client initialization
Fix issue with --multi option in console log collection for Chrome browser
Fix promise handling for objects that are not promises but have a `then` method
Update deployAll method in Test class to use simulator provider
Update TV endpoint for Skjarinn channel
Added support for filtering requests based on HTTPS server and port.
Improve performance of JavaScript code hints by optimizing token lookups
Update range parsing tests with new constants and edge cases
Fix grid parsing error and add data property to grid object
Added onEnterKeyPress prop and updated isValid prop type for better validation handling
Added functionality to ignore a folder in Favorites and navigate back to the previous screen after ignoring it.
Added support for retrieving translations for labels based on locale
Update order.js with new resource and geopoint
Update TimelineSignal to use raw values instead of converted values
Added findOne method to MongoDB service for retrieving a single document from a collection.
Update form input mixin to pass event object as second argument to onFocus and onBlur callbacks
Improve graceful shutdown of worker threads by using 'done' and 'cancelled' messages instead of 'exit'.
Added empty() and css() methods to hide and clear the contents of the section-trees and section-actions elements.
Update PhantomJS dependency to use prebuilt version instead of system installed one.
Added support for multiple colors in smartFlatColor function and updated tests accordingly.
Added support for Mixed data type in AllDataElements schema
Improve performance by reducing unnecessary requests to the server when searching for options in Select2's ajax mode.
Added support for nested objects in schemaForm
Update memcached spawn options to include a 10 minute idle timeout (-I 10m) for improved performance and memory usage.
Added support for watch mode in webpack configuration
Added event listener for content_clicked.slidingtree to close the action bar when a content is clicked.
Update useTransition hook to support custom from values for each item in a transition
Fix error in bitz.js file by removing unnecessary console.error() call
Update Stamen tile URL to use SSL and Fastly CDN
Fix duplicate page URLs in TemplateMap
Sync a doc
Update SnackbarContent styles to use color manipulation for better contrast
Added placeholder text color prop for customization of input field placeholder text color
Improve pagination performance by only calculating the width of the tabs when necessary.
Added log levels for outputting to std.out() (ex. log.v(), log.d() ...)
Refactor deprecated ContainerView and add new ContainerView class for Ember 3.x compatibility
Added logging functionality for debugging purposes
Removed unused dependency on jquery-catchpaste library in Gruntfile.js
Added function 'isForced' to check if a sync is forced or not. This function takes a path as an argument and returns a promise that resolves to true if the sync is forced for that path, false otherwise.
Update moderation API endpoint for whitelisting users
Added polyfill for requestAnimationFrame and clearTimeout functions to support older browsers.
Added test for put with _deleted: true
Update filter-repositories controller to display correct number of selected repositories
Update icon mapping for note tag to use more consistent icons
Update decision view priority to 90
Refactor code to improve performance and reduce memory usage by removing unnecessary event listeners and optimizing the stream handling.
Update module exports for new version of EE library
Update turbolinks_events.js to set flag for loading UV script explicitly
Refactor export statements for easier import syntax
Fix directory node removal in ProjectManager.js
Update offset calculation for Chrome 45-46+
Added keyboard shortcut for menu toggle (/) when not in an input field.
Fix incorrect key names in Chrome for digit keys with modifiers
Update admin filter state and update users in list
Update U2F library to use latest crypto module and remove unused challenge generator module.
Fix for IE10 compatibility issue with uniqueId_ generation
[Babel] Update webpack config to use babel-loader instead of babel
Update GitHub login method to use new access token
Update navigation script to handle changes in link and history state
Update input stories to use new form component
Added functionality to modify the raw theme's zIndex and fontFamily properties. Added a new method 'modifyRawThemePalette' to recompute the MUI theme based on the new raw theme.
Update StaticServlet to handle URL encoding correctly
Enable debug functionality in rollup.config.js for browser bundle
Fix Zip function time logging
Update regex pattern for matching index files in jsx-classname-namespace rule
Update Chatroom Tests: remove unnecessary test cases and simplify existing ones
Update tests for Comlink in the same realm
Fix incorrect require path for webidl2 library in test/invalid.js
Update discussion list rendering to only display when navigation is not visible.
Added significant precision flag for Bitfinex exchange
Added support for async parsing and improved error handling in sqliteParser function
Added strict mode and refactored code to use ES6 class syntax for MultiWatching class.
Update editor mapping for Rubymine
Update node.test.js to remove top style and add test for empty string
Update API configuration to use temporary directory for locale files
Update movie metadata from Cinemeta API
Update configuration merge function to handle objects correctly
Added default export for TAP class and updated module exports to include both the TAP class and the default export.
Added support for inclusive date range validation using `inclusivity` option in `between()` method of `avDate` schema
Update hermione configuration for increased stdout buffer size
Update gitPlugin.js to use new Git API for computing diff
Added support for custom dash patterns in line charts
Added support for input pagination
Refactor keyframe sorting logic to improve performance
Fix issue with canceling UR during AirService test
Update build type for testing
Added error handling for latest task in runner function. Added console log and process exit with status 1 if latest task fails.
Refactor useQueries to deprecate query argument in createPath and createHref methods
Fix duplicate URL generation bug by retrying on error code 11000
Update downloadBitbucket.js to handle response data as a string and parse it as JSON
Update teamwork class to handle custom domains with protocol and without trailing slash
Added support for multiple connections in Hapi server
Reordered attributes in Element class to ensure 'type', 'max', and 'min' are first, followed by 'value'. This change was made to improve the readability of the code.
Added support for setting multiple values for headers using an array
Update public resolver address in ensutils.js
Update migrations directory resolution logic to use default when no migrationsDir is specified in config file or unable to read config file
Added mouseout event listener to color picker dialog to restore previous hovered cell on mouse out.
Added support for running Chrome in a virtual framebuffer (Xvfb) using the --xvfb option. This allows Chrome to run without a display and is useful for headless testing.
Lazy load fs.notify and findit dependencies for watch API tests
Added support for timeout parameter in Aástar pathfinding algorithm
Removed unnecessary export of twig module and updated exports for better compatibility with other libraries.
Update is.equal function to handle strict equality check for Date objects
Update elasticsearch URL to use HTTPS and include port number if present
Refactor: Remove private routes from protectedRoutes list
Added tests for Merge function's behavior with different inputs and outputs.
Update getCdps schema to include proxy address in the id field
Added support for TAP junit output format in the writeOutput function
Update directive binding and unbinding logic to support Vue 3.x
Fix uncaught error when processing an image with bugos resize value
Update NodeTestHelper to start flows after loading flows
Update tilelive auto module to support multiple formats
Refactor Drawer component to use RestrictFocusToChildren instead of MXFocusTrap
Added support for debugging in TinkerHub.js library.
Update Orvibo device model to include power source and battery percentage remaining
Added support for sending chat messages with a callback function
Refactored graph.js to use a single object for node attributes instead of separate functions.
Update require config for extra packages in jam.json files
Update changelog with explicit log file
Fix issue with clearing database in production mode
Update Stripe API version in service
Refactor code to use a single callback function for all cases of setting and getting code.
Refactoring of reveal.js initialization for PDF printing
Fix trailing slash in handle-image-sizes middleware
Added compatibility for LazyFiltersResult in compatibilityValuesRamp function
Update mouse mode selector event data with deltaX and deltaY for better precision in mouse movement tracking.
Added support for updating the navigation bar based on props changes
Refactored deserializer to use a single handler map instead of separate handlers for each content type.
Added support for generating service specs based on project configuration
Added a debugger statement to PlaybackState class to help with debugging.
Update import statements in test/library.spec.js to use the new module structure
Update electron build tasks for armv7l architecture
Refactor slider component to use new internal transitioner module
Fix issue with special character links in Nightwatch tests
Fix ESC key handling in dialogs to execute action instead of closing them.
Added support for playback rate and quality level retrieval
Added new components for user and dumb component examples
Added support for full screen modals and modal cards
Added support for Mosaic Layer type in ArcGIS catalog group model
Update Tessel.prototype.rename to use MAC address instead of name
Update selector function to support empty query and wildcard search
Added password prompt for changing user password
Remove file method added to nodeStatics.js module
Update Item style to use marginTop instead of paddingTop
Added support for specifying a map path directly in createEngine() function instead of relying on findMap() function.
Update HTTPS certificate and key for unexpected-messy test
Fix JSON response parsing issue in ccm_parseBlockResponse() function
Fix bug in NWMatcher.js where native getElementById() was not working correctly with names.
Update annotations file for LiveReload performance improvement
Added support for rendering React components in an iframe using ReactDOM.findDOMNode() and ReactDOM.render().
Added support for CUI Modal in editor-spec.js
Update timeago directive for Spanish language support. Fixes #1234.
Added tests for handling charsets in content types. (#1234)
Update comment loader template to use new comment name variable
Added support for custom error handling in server.raise() method
Stop drones and remove them from the application's list of drones. Fix #1234.
Update password for an existing account
Fix invalid template error when using custom templates
Fix bug in loggly client integration by logging single request instead of multiple requests.
Fix nginx startup issue by ignoring error log file open failure and setting started flag when worker process starts.
Refactor utils for well naming and key conversion. Add unit tests for new utils.
Fix for issue where empty string was being added to the document when pressing enter in an empty line.
Fix promise handling in superagent-promise-plugin
Added support for fading between two images in the image node. Added a new input for the fade value and updated the output type to image.
Added test for duplicate import paths in CSS files
Fix ElasticSearch index error handling
Update property definition for entity <entity> in schema.js
Added toggleUi option for schema widgets UI
Refactor button rendering to improve performance and fix a bug related to visibility detection. (#1234)
Fix formatting of buffer generation code
Update default settings for exposing tasks in the task registry. Add new setting `exposeWithPrefix` with value `'auto'`. Remove setting `exposeWithoutPrefix`. Fix typo in log statement.
Refactor Bloom filter implementation to use safe-buffer instead of Buffer for compatibility with Node.js 10 and later versions.
Update realpathSync and realpath native methods to use fs.realpathSync and fs.realpath respectively.
Refactor gulp build process to use browserify and bundle-collapser plugins instead of streamify and vinyl-source-stream. Remove unnecessary dependencies.
Fix save error handling and add a tip for too large bins
Refactor AutoCompleteList component to use new props API
Update environment variables and settings loading mechanism
Update webpack configuration to disable CSS URL processing for Honey Comb admin panel
Added support for installing GitHub workflows in Yarn Berry projects
Update input focus for OnyxMatters app
Added function parseCookies to parse cookies from a request object
Fix issue with spinner not being hidden after form submission
Fix issue with DOM not being ready on startup in non-NodeJS environments
Adjust heights on didUpdate ember event
Update modal max height calculation to account for margin and padding.
Fix year mapping in createDate function to ensure correct behavior for years between 0 and 99.
Added support for displaying available options in the network selector when pressing Ctrl+Shift+Alt+A.
Added lodash dependency for query editor component
Fix issue with external value change and masking
Update regex for splitting file paths
Added event listeners for focus and blur events to optimize notification polling
Update ads.js to include client ID for Google AdSense
Update player physics to use THREE.Vector3 instead of game.THREE.Vector3
Added support for customizing card headers with `rightHeader` prop
Update search functionality to use async/await syntax
Added support for providing own promise implementation using `promise()` method.
Added support for missing server-to-client packet processor (103) and updated client code to handle new packet type (240).
Fix issue with layer click event not firing when dragging a marker on top of it
Added blur event listener for ContentEditableBinding to improve performance and fix a bug where the value was not updated when the element lost focus.
Added timestamp and datetime fields to the response object in the aofex.js file.
Added support for reading file streams and added length property to file metadata
Update profiler defaults for Firefox. Increase buffer size from 100MB to 13107200B (100MB).
Refactor structure index to use sets instead of objects for better performance and memory usage.
Fix issue with empty function names in instruments.js
Added check for window object to prevent server-side rendering issue. See #22.
Fix viewport size calculation for active area computation
Add support for struct definitions in contract source parsing
Added focus() method to contentWindow object to ensure that the webview has focus when it is created.
Update playmodes in Sonos.prototype.getPlayMode() to include SHUFFLE_NOREPEAT mode.
Update flag plugin to handle response with missing data
Added support for copying a Loki database with a new environment detection fallback
Added support for Map and Set data structures in IE11 by requiring core-js modules.
Added support for multiple STOMP clients with unique IDs and debug logging
Fix string parsing issue in stream() function
Added support for parsing ES6 syntax in test_filter.js
Update html-parser.js to use elenpi.htmlParser instead of this.htmlParser
Update file index.js to handle errors in input and output streams
Remove MongoDB specific code for migrations
Added support for reading and writing multiple formats using the `readers` and `writers` objects.
Update check-version-md script to skip prerelease versions and log a warning message when a prerelease version is detected.
Update authenticationSource.js viewFields for sources text
Added support for specifying a controller by name in the `{{render}}` helper.
Update minimumInputLength for XREF and MREF fields in jquery.molgenis.xrefmrefsearch.js
Added new component 'Column' to siteIndex.js
Added support for hash location type in dummy app config file
Fix issue with GDAX API request for candlestick data. The 'end' parameter was set incorrectly, causing the API to return an empty response. This has been fixed by adjusting the value of the 'end' parameter to match the expected format.
Update TCL mode to support new syntax features and fix bugs
Refactor BlockchainWallet class to use ES6 syntax and add support for second password.
Added support for Firefox and fixed positioning on IE and FireFox to avoid glitches in parallax effect.
Update distribution directory path in getArtifactMtimes function
Fix output path for nightwatch plugin
Update npm registry check for latest version of plugin
Improve performance of asyncjs by using requestAnimationFrame or webkitRequestAnimationFrame if available, and fallback to timeout if not.
Fixes issue with HTML comments in content by removing them and updating the regex for finding files.
Update timestamp formatting to account for timezone offset
Added support for custom SVGSpriteLoaderConfig via query parameter
Added support for caching VAOs in GeometrySystem.js
Added support for global document object in Delegator constructor
Added pastebin feature to debugger.js
Update Coroutine resume method to handle dead coroutines
Update Chance library import to use new constructor syntax
Update ember-flexberry-data to include register-version.js in the vendor tree.
Added polyfill for Element.prototype.cloneNode() detection
Refactor debounce function to use requestAnimationFrame for better performance and cancelable debouncing
Update PID controller constants and remove dispKp, dispKi, dispKd properties
Added JSON.stringify() for data parameter in sendWebPush function to ensure that it is a string before sending it to web-push library
Added polyfill for Date.now() in IE8 and below
Enable combo api and disable sync loader in dojo/_loaderExt.js
Added protection for vTree in attributeChanged event handler
Update file upload functionality to handle non-string HTML values
Fix XMLHttpRequest error handling and notifyActivity() calls
Fix issue with dropping columns in SQLite3 dialect by updating the regex pattern used to identify the column to drop.
Added support for setting pixel ratio and updated tile size calculation based on device pixel ratio.
Update codemirror mode for HTML mixed with CSS and JavaScript
